import os
import numpy as np
import pandas as pd
import traceback
from dotenv import load_dotenv
import joblib

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import smape, mae, mse
from darts.models import TSMixerModel
from darts.dataprocessing.transformers import MissingValuesFiller
from darts.utils.likelihood_models import QuantileRegression

# Data processing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Visualization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Model evaluation
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay

import wandb
wandb.init(settings=wandb.Settings(start_method='thread'))


if __name__ == "__main__":

    load_dotenv()
    DATASET = os.getenv("OUTPUT_DIR_v2")

    # load dataset
    dataset = pd.read_csv(DATASET)

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

    # Convert DataFrame into Darts TimeSeries format
    full_target_ts = TimeSeries.from_dataframe(dataset, value_cols=target_cols, time_col=time_column_name, fill_missing_dates=True)
    full_covariates_ts = TimeSeries.from_dataframe(dataset, value_cols=covariate_cols, time_col=time_column_name, fill_missing_dates=True)

    # Fill NaNs that might have been created
    filler = MissingValuesFiller()
    full_target_ts = filler.transform(full_target_ts)
    full_covariates_ts = filler.transform(full_covariates_ts)

    # Define split proportions
    target_past_ratio = 0.31  # 30% for training
    target_true_ratio = 0.10   # 10% for validation
    target_past_start_idx, target_past_end_idx = 22050, 22050+3200
    target_true_end_idx = target_past_end_idx+576

    target_past_split_point = int(len(full_target_ts) * target_past_ratio)
    target_true_split_point = int(len(full_target_ts) * (target_true_ratio + target_past_ratio))

    # Assign to global variables for Optuna's objective function
    
    target_past_ts = full_target_ts[target_past_start_idx:target_past_end_idx]
    target_true_ts = full_target_ts[target_past_end_idx:target_true_end_idx]

    print(f"FULL Target TS Length: {len(full_target_ts)}")
    print(f"Target Past TS Length: {len(target_past_ts)}")
    print(f"Target True TS Length: {len(target_true_ts)}")

    covariates_past_ts_unscaled = full_covariates_ts[target_past_start_idx:target_past_end_idx]

    # Scale Covariates
    pump_scaler = joblib.load("darts_models/TSMixer_Final_Trained_Model/pump_covariate_scaler.pkl")

    train_past_covariates_scaled = pump_scaler.transform(covariates_past_ts_unscaled)

    if train_past_covariates_scaled: print(f"Scaled Train Covariates shape: {train_past_covariates_scaled.to_dataframe().shape}")
    

    # Load the model
    model = TSMixerModel.load_from_checkpoint(
        model_name="TSMixer_Final_Trained_Model", 
        work_dir="./darts_models", 
        best=True
        )

    print(f"Model loaded: {model}")

    output_chunk_length = 576
    preds_ts = model.predict(
        n=output_chunk_length, # Output length
        series=target_past_ts, # (train + val target)
        past_covariates=train_past_covariates_scaled, # (train + val covariates)
        num_samples=200,
    )

    # Get the median TimeSeries using the correct method name
    median_pred_ts = preds_ts.quantile_timeseries(quantile=0.50)

    # Extract the numpy array from that TimeSeries
    pred_scores_per_class = median_pred_ts.values()

    # Get the predicted class labels (this logic was already correct)
    predicted_class_labels = np.argmax(pred_scores_per_class, axis=1).flatten()

    # Get actual labels from the validation set
    actual_target_matrix = target_true_ts.values(copy=True)
    actual_class_labels = np.argmax(actual_target_matrix, axis=1).flatten()

    # Check lengths before scoring
    if len(actual_class_labels) != len(predicted_class_labels):
        print(f"FATAL: Mismatch in prediction/actual lengths. Actual: {len(actual_class_labels)}, Pred: {len(predicted_class_labels)}")
    
    # Ensure lengths match for evaluation
    min_eval_len = min(len(actual_class_labels), len(predicted_class_labels))

    if min_eval_len < 1: # Need at least one point to evaluate
        print(f"Warning: Prediction length ({min_eval_len}) on validation set is too short.")

    actual_class_labels = actual_class_labels[:min_eval_len]
    predicted_class_labels = predicted_class_labels[:min_eval_len]

    print(f"Actual class labels: {actual_class_labels}")
    print("\n ----------------------------------------------- \n")
    print(f"Predicted class labels: {predicted_class_labels}")

    # roc_auc = roc_auc_score(actual_class_labels, predicted_class_labels, multi_class='ovr', average='weighted', labels=[0, 1, 2])

    # print("\n--- Evaluation Metrics ---")
    # print(f"ROC AUC Score: \n {roc_auc:.4f}")
    # print("\n --------------------------------------- ")

    label_map = {
        0: 'BROKEN',
        1: 'NORMAL',
        2: 'RECOVERING'
    }

    report = classification_report(actual_class_labels, predicted_class_labels, 
                                    zero_division=0, output_dict=True, labels=[0, 1, 2], 
                                    target_names=["BROKEN", "NORMAL", "RECOVERING"])
    
    print("\n--- Classification Report ---")
    print(report)
    print("\n --------------------------------------- ")

    unique_predicted_labels = np.unique(predicted_class_labels)
    target_names = ['BROKEN', 'NORMAL', 'RECOVERING']
    target_labels = [0, 1, 2]

    cm = confusion_matrix(actual_class_labels, predicted_class_labels, labels=target_labels)
    fig_cm, ax_cm = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax_cm, cmap=plt.cm.Blues, values_format='d')
    ax_cm.grid(False)
    ax_cm.set_title(f'Confusion Matrix')
    plt.show()