import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Preprocessing
from feature_engine.selection import RecursiveFeatureElimination

# Training
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping as PLEarlyStopping

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TSMixerModel
from darts.utils.likelihood_models import QuantileRegression

# Data processing
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Model evaluation
from sklearn.metrics import roc_auc_score

# Hyperparameter tuning
import optuna
from optuna.integration import PyTorchLightningPruningCallback

'''
Best params: {
'in_len': 493, 
out_len': 120,
'hidden_size': 32, 
'ff_size': 112, 
'num_blocks': 2, 
'dropout': 0.24602056233523145, 
'lr': 1.4216953764344364e-05, 
'activation': 'GELU', 
'batch_size': 32}
'''

# --- Global variables for Optuna study ---
train_target_ts = None
val_target_ts = None
test_target_ts = None # This is the set Optuna will evaluate against

train_past_covariates = None
val_past_covariates = None
test_past_covariates_scaled = None # Covariates for the test_target_ts (Optuna eval set)

# define objective function
# Define objective function for Optuna
def objective(trial: optuna.trial.Trial) -> float:

    # Hyperparameters to tune for TSMixerModel
    in_len = trial.suggest_int("in_len", 360, 600)
    out_len = 120  
    hidden_size = trial.suggest_int("hidden_size", 32, 256, step=8)
    ff_size = trial.suggest_int("ff_size", 64, 256, step=16) # Feed-forward layer size
    num_blocks = trial.suggest_int("num_blocks", 1, 4)    # Number of TSMixer blocks
    dropout = trial.suggest_float("dropout", 0.05, 0.3)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    activation = trial.suggest_categorical("activation", ["ReLU", "LeakyReLU", "GELU"])
    batch_size = trial.suggest_categorical("batch_size", [16, 32])

    # Data length checks
    if len(train_target_ts) < in_len + out_len or \
       len(val_target_ts) < out_len or \
       len(test_target_ts) < out_len : # Ensure test set is also long enough
        print(f"Trial {trial.number} pruned: Data length insufficient.")
        raise optuna.exceptions.TrialPruned()

    # Callbacks for PyTorch Lightning
    # Monitor "val_loss" which Darts models compute on their validation set
    pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    early_stopper = PLEarlyStopping("val_loss", min_delta=0.001, patience=20, verbose=False) # Reduced patience
    callbacks = [pruner, early_stopper]

    pl_trainer_kwargs = {
        "accelerator": "auto", # Uses GPU if available, else CPU
        "callbacks": callbacks,
        "enable_progress_bar": True,
    }

    torch.manual_seed(42) # For reproducibility per trial

    # Build the TSMixer model
    model = TSMixerModel(
        input_chunk_length=in_len,
        output_chunk_length=out_len,
        hidden_size=hidden_size,
        ff_size=ff_size,
        num_blocks=num_blocks,
        batch_size=batch_size,
        activation=activation,
        n_epochs=300, # Max epochs; early stopping will manage the actual number
        nr_epochs_val_period=1, # How often to check validation loss
        dropout=dropout,
        optimizer_kwargs={"lr": lr},
        likelihood=QuantileRegression(), # For probabilistic forecasts -> ROC AUC on median
        pl_trainer_kwargs=pl_trainer_kwargs,
        model_name=f"TSMixer_Pump_Trial_{trial.number}",
        force_reset=True, # Ensures fresh model per trial
        save_checkpoints=False, # No need to save checkpoints during Optuna
    )

    try:
        # Train the model
        # Darts models use their own internal validation split during .fit() if val_series is provided
        model.fit(
            series=train_target_ts,
            val_series=val_target_ts, # Darts uses this for early stopping and val_loss
            past_covariates=train_past_covariates,
            val_past_covariates=val_past_covariates, # Covariates for the validation set
            verbose=False
        )

        probabilistic_preds_val = model.predict(
            n=len(test_target_ts),
            series=eval_target_ts, # History of target to condition on
            past_covariates=eval_past_covariates_scaled  # Covariates for the validation period
        )

        # Extract median for point forecast since QuantileRegression is used
        preds_val_median = probabilistic_preds_val.quantile_timeseries(quantile=0.5)

        actual_target_vals = test_target_ts.values(copy=True).flatten()
        pred_continuous_val = preds_val_median.values(copy=True).flatten()

        # Ensure lengths match for evaluation
        min_eval_len = min(len(actual_target_vals), len(pred_continuous_val))

        if min_eval_len < 1:
            print(f"Warning: Prediction length ({min_eval_len}) on validation set is too short for trial {trial.number}.")
            return float('inf') 

        actual_vals_eval = actual_target_vals[:min_eval_len]
        pred_continuous_eval = pred_continuous_val[:min_eval_len]

        print("actual_vals_eval:", actual_vals_eval[:5])
        print("len(actual_vals_eval):", len(actual_vals_eval))
        print("Uniqeu : ", np.unique(actual_vals_eval))
        print("Length Unique: ", len(np.unique(actual_vals_eval)))

        # Check if validation target has both classes for ROC AUC
        if len(np.unique(actual_vals_eval)) < 2:
            print(f"Warning: Trial {trial.number} - ROC AUC cannot be calculated. Validation target has only one class. Returning worst score (1.0).")
            # This can happen if a small validation set doesn't contain both 0s and 1s
            return 1.0 # Return a "bad" score for Optuna to minimize

        # Calculate ROC AUC score
        score = roc_auc_score(actual_vals_eval, pred_continuous_eval)
        objective_value = 1.0 - score # Optuna minimizes, so we minimize (1 - AUC)

        if np.isnan(objective_value) or np.isinf(objective_value):
            print(f"Warning: Objective value is NaN/inf for trial {trial.number}. ROC AUC: {score}. Returning inf.")
            return float('inf') # Should not happen with prior checks

        return objective_value

    except optuna.exceptions.TrialPruned:
        raise # Re-raise for Optuna to handle
    except RuntimeError as e: # Catch PyTorch/CUDA errors
        if "CUDA out of memory" in str(e) or "Address already in use" in str(e):
            print(f"Optuna trial {trial.number} failed with RuntimeError (likely CUDA OOM or port issue): {e}. Pruning.")
            raise optuna.exceptions.TrialPruned() # Prune if it's a resource issue
        print(f"Optuna trial {trial.number} failed with RuntimeError: {e}")
        return float('inf') 
    except ValueError as e: 
        print(f"Optuna trial {trial.number} - ValueError: {e}. Returning bad score (1.0).")
        return 1.0
    except Exception as e: # Catch any other unexpected errors
        print(f"Optuna trial {trial.number} failed with an unexpected error: {e}")
        return float('inf')


# for convenience, print some optimization trials information
def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")


if __name__ == "__main__":

    load_dotenv()
    DATASET = os.getenv("OUTPUT_DIR")

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
    # dataset.sort_values(by='timestamp', inplace=True)

    # --- Data Preparation for Fine Tuning ---
    time_column_name = 'timestamp' 
    target_column_name = 'machine_status'
    covariate_cols = [col for col in dataset.columns if col not in [target_column_name, time_column_name]]

    print(f"\nTarget column: ['{target_column_name}']")
    print(f"Covariate columns ({len(covariate_cols)}): {covariate_cols[:5]}...")
    print(f"Time column: '{time_column_name}'")

    data_freq = 'min'

    # --- Convert DataFrame into Darts TimeSeries format ---
    full_target_ts = TimeSeries.from_dataframe(dataset, value_cols=[target_column_name], time_col=time_column_name, fill_missing_dates=True, freq=data_freq)
    full_covariates_ts = TimeSeries.from_dataframe(dataset, value_cols=covariate_cols, time_col=time_column_name, fill_missing_dates=True, freq=data_freq)

    # Define split proportions
    train_frac = 0.7  # 70% for training
    val_frac = 0.15   # 15% for validation

    val_split_point = int(len(full_target_ts) * train_frac)
    test_split_point = int(len(full_target_ts) * (train_frac + val_frac)) # End of validation, start of test

    # Assign to global variables for Optuna's objective function
    train_target_ts = full_target_ts[:val_split_point]
    val_target_ts = full_target_ts[val_split_point:test_split_point]
    eval_target_ts = full_target_ts[:test_split_point]
    test_target_ts = full_target_ts[test_split_point:] # This is test_target_ts_for_optuna_eval

    train_covariates_ts_unscaled = full_covariates_ts[:val_split_point]
    val_covariates_ts_unscaled = full_covariates_ts[val_split_point:test_split_point]
    eval_covariates_ts_unscaled = full_covariates_ts[:test_split_point]
    test_covariates_ts_unscaled = full_covariates_ts[test_split_point:]
    
    # --- Scale Covariates ---
    covariate_scaler = Scaler(StandardScaler())
    train_past_covariates = covariate_scaler.fit_transform(train_covariates_ts_unscaled)
    val_past_covariates = covariate_scaler.transform(val_covariates_ts_unscaled)
    eval_past_covariates_scaled = covariate_scaler.fit_transform(eval_covariates_ts_unscaled)
    test_past_covariates_scaled = covariate_scaler.transform(test_covariates_ts_unscaled) 

    print(f"\nData split lengths: Train Target={len(train_target_ts)}, Val Target={len(val_target_ts)}, Test Target (for Optuna)={len(test_target_ts)}")
    if train_past_covariates: print(f"Scaled Train Covariates shape: {train_past_covariates.to_dataframe().shape}")
    if val_past_covariates: print(f"Scaled Val Covariates shape: {val_past_covariates.to_dataframe().shape}")
    if test_past_covariates_scaled: print(f"Scaled Test Covariates shape: {test_past_covariates_scaled.to_dataframe().shape}")


    # --- Optuna Hyperparameter Tuning ---
    n_optuna_trials = 50
    print(f"\n--- Starting Optuna Hyperparameter Tuning ({n_optuna_trials} trials) ---")
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_optuna_trials, callbacks=[print_callback])
  