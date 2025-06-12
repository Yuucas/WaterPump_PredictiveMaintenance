import os
import numpy as np
import pandas as pd
import traceback
from dotenv import load_dotenv

# Preprocessing
from feature_engine.selection import RecursiveFeatureElimination

# Training
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import smape, mae, mse
from darts.models import TSMixerModel
from darts.utils.likelihood_models import QuantileRegression
from darts.dataprocessing.transformers import MissingValuesFiller

# Data processing
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Visualization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Model evaluation
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay

# Tracking
import wandb



if __name__ == "__main__":

    load_dotenv()
    DATASET = os.getenv("OUTPUT_DIR_v2")

    # load dataset
    dataset = pd.read_csv(DATASET)
    print("Dataset head: \n", dataset.head())

    # Description of dataset
    print("Description of dataset: \n", dataset.describe())

    # the data types
    print("Data Types: \n", dataset.info())

    # Check for missing values
    print(" Missing Values: \n", dataset.isna().sum())

    # Check Type counts
    print("Check Type Counts: \n", dataset['machine_status'].value_counts())

    # Convert timestamp and sort
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp'])
    dataset.sort_values(by='timestamp', inplace=True)

    # Data Preparation for Fine Tuning
    time_column_name = 'timestamp' 
    target_column_name = 'machine_status'

    ohe = OneHotEncoder(sparse_output=False, categories=[[0, 1, 2]])

    # Fit on the entire dataset to ensure all categories are known
    encoded_targets = ohe.fit_transform(dataset[[target_column_name]])
    encoded_df = pd.DataFrame(encoded_targets, columns=ohe.get_feature_names_out([target_column_name]), index=dataset.index)

    # Join back with the original dataframe
    dataset = dataset.join(encoded_df)

    # Update target and covariate column lists
    target_cols = ohe.get_feature_names_out([target_column_name]).tolist()
    covariate_cols = [col for col in dataset.columns if col not in target_cols + [target_column_name, time_column_name]]

    print(f"\nTarget column: ['{target_cols}']")
    print(f"Covariate columns ({len(covariate_cols)}): {covariate_cols[:5]}...")
    print(f"Time column: '{time_column_name}'")

    # Convert DataFrame into Darts TimeSeries format
    full_target_ts = TimeSeries.from_dataframe(dataset, value_cols=target_cols, time_col=time_column_name, fill_missing_dates=True)
    full_covariates_ts = TimeSeries.from_dataframe(dataset, value_cols=covariate_cols, time_col=time_column_name, fill_missing_dates=True)

    # Fill NaNs that might have been created
    filler = MissingValuesFiller()
    full_target_ts = filler.transform(full_target_ts)
    full_covariates_ts = filler.transform(full_covariates_ts)

    # Define split proportions
    train_frac = 0.82  # 84% for training
    val_frac = 0.09   # 8% for validation

    val_split_point = int(len(full_target_ts) * train_frac)
    test_split_point = int(len(full_target_ts) * (train_frac + val_frac)) # End of validation, start of test

    # Assign to global variables for Optuna's objective function
    train_target_ts = full_target_ts[:val_split_point]
    val_target_ts = full_target_ts[val_split_point:test_split_point]
    test_target_ts = full_target_ts[test_split_point:]
    eval_target_ts = train_target_ts.append(val_target_ts)

    train_covariates_ts_unscaled = full_covariates_ts[:val_split_point]
    val_covariates_ts_unscaled = full_covariates_ts[val_split_point:test_split_point]
    test_covariates_ts_unscaled = full_covariates_ts[test_split_point:]
    
    # Scale Covariates
    covariate_scaler = Scaler(StandardScaler())

    train_past_covariates_scaled = covariate_scaler.fit_transform(train_covariates_ts_unscaled)
    val_past_covariates_scaled = covariate_scaler.transform(val_covariates_ts_unscaled)
    test_past_covariates_scaled = covariate_scaler.transform(test_covariates_ts_unscaled)
    eval_covariates_scaled = train_past_covariates_scaled.append(val_past_covariates_scaled)

    print(f"\nData split lengths: Train Target={len(train_target_ts)}, Val Target={len(val_target_ts)}, Test Target (for Optuna)={len(test_target_ts)}")
    if train_past_covariates_scaled: print(f"Scaled Train Covariates shape: {train_past_covariates_scaled.to_dataframe().shape}")
    if val_past_covariates_scaled: print(f"Scaled Val Covariates shape: {val_past_covariates_scaled.to_dataframe().shape}")
    if test_past_covariates_scaled: print(f"Scaled Test Covariates shape: {test_past_covariates_scaled.to_dataframe().shape}")


    # Best parameters from Optuna
    best_hyperparams = {
        'input_chunk_length': 3200,
        'output_chunk_length': 576,
        'hidden_size': 112,
        'ff_size': 64,
        'num_blocks': 4,
        'dropout': 0.2307908071260658,
        'optimizer_kwargs': {'lr': 0.005283329371057148},
        'activation': 'LeakyReLU'
        # 'MaxPool1d': False 
    }

    # --- Initialize W&B Logger for PyTorch Lightning ---
    model_name = "TSMixer"
    wandb_project_name = "water-pump_pm" # Define your project name
    wandb_run_name = f"{model_name}_Final_in{best_hyperparams['input_chunk_length']}_lr{best_hyperparams['optimizer_kwargs']['lr']:.0e}"

    wandb_logger = WandbLogger(
        project=wandb_project_name,
        name=wandb_run_name,
        job_type="final-model-training"
    )


    # --- Final Model Parameters ---
    final_model_params = {
        'output_chunk_length': best_hyperparams['output_chunk_length'],
        'batch_size': 16,
        'n_epochs': 200,
        'nr_epochs_val_period': 1,
        'likelihood': QuantileRegression(),
        'model_name': f"{model_name}_Final_Trained_Model", # local checkpoints
        'work_dir': './darts_models', 
        'force_reset': True,
        'save_checkpoints': True,
        'random_state': 42
    }

    # Merge Optuna best params with other fixed params
    final_model_params.update(best_hyperparams)

    # Log final model configuration to W&B (WandbLogger does this if config is passed to it,
    wandb_logger.experiment.config.update(final_model_params)


    # --- Callbacks and Trainer Arguments ---
    final_early_stopper = EarlyStopping("val_loss", min_delta=0.005, patience=15, verbose=True, mode="min") # Increased patience
    final_callbacks = [final_early_stopper]

    final_pl_trainer_kwargs = {
        "accelerator": "cuda" if torch.cuda.is_available() else "cpu",
        "callbacks": final_callbacks,
        "enable_progress_bar": True,
        "logger": wandb_logger # Integrate WandbLogger
    }
    final_model_params['pl_trainer_kwargs'] = final_pl_trainer_kwargs

    # Create and train the final model
    final_model = TSMixerModel(**final_model_params)

    print(f"\n--- Training Final {model_name} Model (Tracked by W&B) ---")

    if len(train_target_ts) < final_model_params['input_chunk_length'] + final_model_params['output_chunk_length'] or \
       (val_target_ts is not None and len(val_target_ts) < final_model_params['output_chunk_length']):
        print(f"Warning: Training or Validation target series is too short. Adjust parameters or data.")
        if wandb.run: wandb.finish(exit_code=1) # Finish W&B run with an error code
    else:
        final_model.fit(
            series=train_target_ts,
            val_series=val_target_ts,
            past_covariates=train_past_covariates_scaled,
            val_past_covariates=val_past_covariates_scaled,
            verbose=True
        )
        print(f"\n--- Final {model_name} Model Training Complete ---")

        # --- Save the Final Model ---
        final_model_save_path = os.path.join(final_model_params['work_dir'], final_model_params['model_name'], "final_best_model")
        final_model.save(final_model_save_path)
        print(f"Final model explicitly saved to: {final_model_save_path}")

        # --- Log Model as W&B Artifact ---
        model_artifact = wandb.Artifact(
            name=final_model_params['model_name'], # Artifact name
            type="model",
            description=f"Final trained {model_name} model after hyperparameter tuning. Best val_loss.",
            metadata=final_model_params # Log model config with artifact
        )
        model_artifact.add_file(final_model_save_path) # Add the saved model file
        wandb_logger.experiment.log_artifact(model_artifact)
        print(f"Logged model artifact '{final_model_params['model_name']}' to W&B.")

        
    # --- PREDICTION (using the final W&B tracked model) ---
    model_to_predict_with = final_model

    if len(test_target_ts) > 0:
        print("\n--- Making Predictions on Test Set with Final W&B-Tracked Model ---")
        try:

            preds_ts = model_to_predict_with.predict(
                n=best_hyperparams['output_chunk_length'], # Output length
                series=eval_target_ts, # (train + val target)
                past_covariates=eval_covariates_scaled, # (train + val covariates)
                num_samples=100,
            )

            # Get the median TimeSeries using the correct method name
            median_pred_ts = preds_ts.quantile_timeseries(quantile=0.5)

            # Extract the numpy array from that TimeSeries
            pred_scores_per_class = median_pred_ts.values()

            # Get the predicted class labels (this logic was already correct)
            predicted_class_labels = np.argmax(pred_scores_per_class, axis=1).flatten()

            # Get actual labels from the validation set
            actual_target_matrix_eval = test_target_ts.values(copy=True)
            actual_class_labels_eval = np.argmax(actual_target_matrix_eval, axis=1).flatten()

            # Check lengths before scoring
            if len(actual_class_labels_eval) != len(predicted_class_labels):
                print(f"FATAL: Mismatch in prediction/actual lengths. Actual: {len(actual_class_labels_eval)}, Pred: {len(predicted_class_labels)}")
            
            # Ensure lengths match for evaluation
            min_eval_len = min(len(actual_class_labels_eval), len(predicted_class_labels))

            if min_eval_len < 1: # Need at least one point to evaluate
                print(f"Warning: Prediction length ({min_eval_len}) on validation set is too short.")

            actual_class_labels_eval = actual_class_labels_eval[:min_eval_len]
            predicted_class_labels = predicted_class_labels[:min_eval_len]

            print(f"\n--- Predicted Results --- \n {predicted_class_labels}")
            print("-------------------------------------")

            # Calculate F1 Score
            score_f1 = f1_score(actual_class_labels_eval, predicted_class_labels, average='weighted', zero_division=0)
            print(f"F1 Score = {score_f1:.4f}")

            accuracy = accuracy_score(actual_class_labels_eval, predicted_class_labels)
            roc_auc = roc_auc_score(actual_class_labels_eval, predicted_class_labels, multi_class='ovr', average='weighted') if len(np.unique(actual_class_labels_eval)) > 1 else 0.5

            wandb_logger.experiment.log({
                "test_accuracy": accuracy,
                "test_roc_auc": roc_auc,
                "test_f1_score": score_f1
            })

            label_map = {
                0: 'BROKEN',
                1: 'NORMAL',
                2: 'RECOVERING'
            }

            report = classification_report(actual_class_labels_eval, predicted_class_labels, 
                                            zero_division=0, output_dict=True, labels=[0, 1, 2], 
                                            target_names=["BROKEN", "NORMAL", "RECOVERING"])
            
            print("\n--- Classification Report ---")
            print(report)

            # --- Logging of Classification Report Metrics ---
            metrics_to_log = {}

            # Log metrics for class 'NORMAL'
            if "NORMAL" in report:
                metrics_to_log["test_precision_class_NORMAL"] = report["NORMAL"]["precision"]
                metrics_to_log["test_recall_class_NORMAL"] = report["NORMAL"]["recall"]
                metrics_to_log["test_f1_class_NORMAL"] = report["NORMAL"]["f1-score"]
                metrics_to_log["test_support_class_NORMAL"] = report["NORMAL"]["support"]
            else: # Log default values if class 'NORMAL' is missing 
                metrics_to_log["test_precision_class_NORMAL"] = 0.0
                metrics_to_log["test_recall_class_NORMAL"] = 0.0
                metrics_to_log["test_f1_class_NORMAL"] = 0.0
                metrics_to_log["test_support_class_NORMAL"] = 0

            # Log metrics for class 'RECOVERING' 
            if "RECOVERING" in report:
                metrics_to_log["test_precision_class_RECOVERING"] = report["RECOVERING"]["precision"]
                metrics_to_log["test_recall_class_RECOVERING"] = report["RECOVERING"]["recall"]
                metrics_to_log["test_f1_class_RECOVERING"] = report["RECOVERING"]["f1-score"]
                metrics_to_log["test_support_class_RECOVERING"] = report["RECOVERING"]["support"]
            else: # Log default values if class 'Failure' is missing
                print("Warning: Class '1' (RECOVERING) not found in classification report. Logging default values for its metrics.")
                metrics_to_log["test_precision_class_RECOVERING"] = 0.0
                metrics_to_log["test_recall_class_RECOVERING"] = 0.0
                metrics_to_log["test_f1_class_RECOVERING"] = 0.0
                metrics_to_log["test_support_class_RECOVERING"] = 0

            # Log metrics for class 'BROKEN' 
            if "BROKEN" in report:
                metrics_to_log["test_precision_class_BROKEN"] = report["BROKEN"]["precision"]
                metrics_to_log["test_recall_class_BROKEN"] = report["BROKEN"]["recall"]
                metrics_to_log["test_f1_class_BROKEN"] = report["BROKEN"]["f1-score"]
                metrics_to_log["test_support_class_RECOVERING"] = report["BROKEN"]["support"]
            else: # Log default values if class 'BROKEN' is missing
                print("Warning: Class '1' (RECOVERING) not found in classification report. Logging default values for its metrics.")
                metrics_to_log["test_precision_class_BROKEN"] = 0.0
                metrics_to_log["test_recall_class_BROKEN"] = 0.0
                metrics_to_log["test_f1_class_BROKEN"] = 0.0
                metrics_to_log["test_support_class_BROKEN"] = 0

            # Log macro averages (these should generally always be present if report is generated)
            if "macro avg" in report:
                metrics_to_log["test_macro_avg_precision"] = report["macro avg"]["precision"]
                metrics_to_log["test_macro_avg_recall"] = report["macro avg"]["recall"]
                metrics_to_log["test_macro_avg_f1"] = report["macro avg"]["f1-score"]
            if "weighted avg" in report: # Also good to log
                metrics_to_log["test_weighted_avg_precision"] = report["weighted avg"]["precision"]
                metrics_to_log["test_weighted_avg_recall"] = report["weighted avg"]["recall"]
                metrics_to_log["test_weighted_avg_f1"] = report["weighted avg"]["f1-score"]


            wandb_logger.experiment.log(metrics_to_log)

            # Get the *full set* of possible labels from the encoder
            all_possible_labels = ohe.categories_[0] 
            # See which labels the model actually predicted
            unique_predicted_labels = np.unique(predicted_class_labels)
            target_names = ['BROKEN', 'NORMAL', 'RECOVERING']
            print(f"\nUnique labels predicted by the model: {unique_predicted_labels}")
            print(f"All possible labels are: {all_possible_labels}")

            # --------------------------------------------------

            cm = confusion_matrix(actual_class_labels_eval, predicted_class_labels, labels=all_possible_labels)
            fig_cm, ax_cm = plt.subplots(figsize=(8, 8))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
            disp.plot(ax=ax_cm, cmap=plt.cm.Blues, values_format='d')
            ax_cm.grid(False)
            ax_cm.set_title(f'Final {model_name} Model - Confusion Matrix')
            wandb_logger.experiment.log({"confusion_matrix": wandb.Image(fig_cm)})
            plt.close(fig_cm)

            # --------------------------------------------------
            # Plotting the actual vs predicted values
            x_min_display, x_max_display = 0, len(predicted_class_labels)-1
            fig_preds, ax_preds = plt.subplots(figsize=(15, 7))

            # Get your full data arrays
            time_index_full_test = test_target_ts.time_index
            actual_labels_np = np.array(actual_class_labels_eval) # Assuming true_labels from previous steps
            predicted_labels_np = np.array(predicted_class_labels) # Assuming pred_labels from previous steps


            # Use direct integer slicing to get the display data
            time_index_display = time_index_full_test[x_min_display:x_max_display]
            actual_labels_display = actual_labels_np[x_min_display:x_max_display]
            predicted_labels_display = predicted_labels_np[x_min_display:x_max_display]

            if len(time_index_display) > 0:
                ax_preds.plot(time_index_display, actual_labels_display, label='Actual Status', marker='o', linestyle='-', color='blue', alpha=0.6, markersize=6)
                ax_preds.plot(time_index_display, predicted_labels_display, label='Predicted Status', marker='x', linestyle='--', color='red', alpha=0.6, markersize=4)
                
                # Update title and labels for clarity
                ax_preds.set_title(f'Final Model: Actual vs. Predicted (Points {x_min_display} to {x_max_display})')
                ax_preds.set_xlabel('Timestamp')
                ax_preds.set_ylabel('Status (0:BROKEN, 1:NORMAL, 2:RECOVERING)')
                ax_preds.set_yticks([0, 1, 2]) # Show all possible status levels
                
                ax_preds.legend()
                ax_preds.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.tight_layout()
                
                # log the figure
                wandb_logger.experiment.log({"actual_vs_predicted_plot": wandb.Image(fig_preds)})

                plt.close(fig_preds) # Close the figure to free up memory

            # --------------------------------------------------

            else: print("Not enough data for evaluation.")
            
        except Exception as e:
            print(f"Error during prediction or evaluation: {e}")
            traceback.print_exc()
    else: print("\nTest set is empty.")

    if wandb.run: # Check if a W&B run is active
         wandb.finish()
    print("\n--- W&B Run Finished (if active) ---")