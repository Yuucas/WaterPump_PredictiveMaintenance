import os
import time
import warnings
from dotenv import load_dotenv

import pandas as pd
import numpy as np
import datetime as dt
from tqdm.notebook import tqdm
from typing import Dict, List, Tuple

import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, precision_recall_fscore_support,
    accuracy_score, f1_score, matthews_corrcoef, 
    confusion_matrix, ConfusionMatrixDisplay
)

from feature_engine.datetime import DatetimeFeatures

# Training
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping as PLEarlyStopping

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TSMixerModel
from darts.utils.likelihood_models import QuantileRegression
from darts.dataprocessing.transformers import MissingValuesFiller


# Data processing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Model evaluation
from sklearn.metrics import f1_score, classification_report, roc_auc_score

# Hyperparameter tuning
import optuna
from optuna.integration import PyTorchLightningPruningCallback

import seaborn as sns
import matplotlib.pyplot as plt


# Generate datetime features
def apply_datetime_features(dataframe, variable_name="Date"):

    dtfs = DatetimeFeatures(
        variables=variable_name,
        features_to_extract=["day_of_year", "month", "hour"],
        drop_original=False
    )

    dfs_transformed = dtfs.fit_transform(dataframe)

    return dfs_transformed


# Define objective function for Optuna
def objective(trial: optuna.trial.Trial) -> float:

    # Hyperparameters to tune for TSMixerModel
    in_len = trial.suggest_int("in_len", 1152, 2016, step=64)
    out_len = 576  
    hidden_size = trial.suggest_int("hidden_size", 32, 256, step=16)
    ff_size = trial.suggest_int("ff_size", 64, 256, step=16) # Feed-forward layer size
    num_blocks = trial.suggest_int("num_blocks", 1, 4)    # Number of TSMixer blocks
    dropout = trial.suggest_float("dropout", 0.01, 0.3)
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    activation = trial.suggest_categorical("activation", ["ReLU", "LeakyReLU", "GELU", "Tanh", "Sigmoid"])
    batch_size = trial.suggest_categorical("batch_size", [16, 32])

    # Data length checks
    if len(train_target_ts) < in_len + out_len or \
       len(test_target_ts) < out_len : # Ensure test set is also long enough
        print(f"Trial {trial.number} pruned: Data length insufficient.")
        raise optuna.exceptions.TrialPruned()
    

    # Callbacks for PyTorch Lightning
    pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    early_stopper = PLEarlyStopping("val_loss", min_delta=0.001, patience=15, verbose=False) # Reduced patience
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
        n_epochs=10, # Max epochs; early stopping will manage the actual number
        nr_epochs_val_period=1, # How often to check validation loss
        dropout=dropout,
        optimizer_kwargs={"lr": lr},
        likelihood=QuantileRegression(), # For probabilistic forecasts -> ROC AUC on median
        pl_trainer_kwargs=pl_trainer_kwargs,
        model_name=f"TSMixer_Pump_Lag_{trial.number}",
        force_reset=True, # Ensures fresh model per trial
        save_checkpoints=False, # No need to save checkpoints during Optuna
    )

    try:
        # Train the model
        # Darts models use their own internal validation split during .fit() if val_series is provided
        model.fit(
            series=train_target_ts,
            past_covariates=train_covariates_ts_scaled,
            val_series=val_target_ts,
            val_past_covariates=val_covariates_ts_scaled,
            verbose=True
        )

        preds_ts = model.predict(
            n=out_len,
            series=input_target_ts, # History of target to condition on
            past_covariates=input_covariates_ts_scaled,  # Covariates for the validation period
            # num_samples=100,  # Number of samples for quantile regression
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
            print(f"Warning: Prediction length ({min_eval_len}) on validation set is too short for trial {trial.number}.")

        actual_class_labels_eval = actual_class_labels_eval[:min_eval_len]
        predicted_class_labels = predicted_class_labels[:min_eval_len]

        # Calculate F1 Score
        score = f1_score(actual_class_labels_eval, predicted_class_labels, average='weighted', zero_division=0)
        print(f"Trial {trial.number}: F1 Score = {score:.4f}")
        
        objective_value = 1.0 - score # We want to minimize the objective, so we return 1 - F1 score

        if np.isnan(objective_value) or np.isinf(objective_value):
            print(f"Warning: Objective value is NaN/inf for trial {trial.number}. F1 Score: {score}. Returning inf.")
            return float('inf')

        return objective_value

    except optuna.exceptions.TrialPruned:
        raise 
    except RuntimeError as e: # Catch PyTorch/CUDA errors
        if "CUDA out of memory" in str(e) or "Address already in use" in str(e):
            print(f"Optuna trial {trial.number} failed with RuntimeError (likely CUDA OOM or port issue): {e}. Pruning.")
            raise optuna.exceptions.TrialPruned() # Prune if it's a resource issue
        print(f"Optuna trial {trial.number} failed with RuntimeError: {e}")
        return float('inf') 
    except ValueError as e: 
        print(f"Optuna trial {trial.number} - ValueError: {e}. Returning bad core (1.0).")
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
    DATA_DIR = os.getenv("RESAMPLED_DATASET")

    # Load the dataset
    dataset = pd.read_csv(DATA_DIR)

    # Remove the 'Unnamed: 0' column if it exists
    if "Unnamed: 0" in dataset.columns:
        dataset.drop(columns=["Unnamed: 0"], inplace=True)
        
    # Get all machine status labels
    all_labels = ['NORMAL', 'RECOVERING']

    # Generate encoding maps for labels
    label2id = {l: i for i,l in enumerate(all_labels)}
    id2label = {v:k for k,v in label2id.items()}
    
    # Replace 'BROKEN' with 'RECOVERING'
    print("--- Before Conversion ---")
    print(dataset['machine_status'].value_counts())
    dataset.loc[dataset['machine_status'] == 'BROKEN', 'machine_status'] = 'RECOVERING'
    print("\n\n--- After Conversion ---")
    print(dataset['machine_status'].value_counts())

    # Encode categorical variables
    label_encoder = LabelEncoder()

    dataset['target'] = label_encoder.fit_transform(dataset['machine_status'])

    # Apply datetime features
    dataset_transformed = apply_datetime_features(dataset, variable_name="timestamp")
    dataset_transformed.head(5)

    # Assign System Failures To Groups
    group_idx = 0
    anomaly_group = [0]
    target_series = dataset_transformed['machine_status'].to_list()

    # Generate groups for each anomaly case
    for _ in range(1, len(dataset_transformed)):
        if (target_series[_] == 'NORMAL' and target_series[_ - 1] == 'RECOVERING'):
            group_idx += 1
            
        anomaly_group.append(group_idx)

    # Add groups to dataset
    dataset_transformed['anomaly_group'] = np.array(anomaly_group).astype(np.int32)

    groups = list(set(dataset_transformed['anomaly_group']))
    group_dict = {}

    # Get system failure instance label counts
    for _ in groups:
        group_dict[_] = dict(
            dataset_transformed[dataset_transformed['anomaly_group'] == _]
            ['machine_status']
            .value_counts()
        )
    # Create summary df for system failure counts    
    group_df = pd.DataFrame(group_dict).T.fillna(0).astype(np.int32)
    print("Grouped DF: \n", group_df.head(5))
    
    # Initialise lag feature variables
    lag_features = []
    n_lags = 5
    feaures_added = False

    # Define list of sensors
    ignored_columns = [
        'timestamp', 'machine_status', 'target', 'timestamp_day_of_year', 'timestamp_month', 'timestamp_hour', 'anomaly_group'
    ]
    sensors = [_ for _ in dataset_transformed.columns if _ not in ignored_columns]

    if not (feaures_added):
        for lag in range(1, n_lags + 1):
            lag_sensors = {}

            for sensor in sensors:
                # Create lag feature name
                sensor_feature = sensor + f'_lag_{lag}'
                
                # Generate lag feature
                lag_feature = dataset_transformed[sensor].shift(lag)
                dataset_transformed[sensor_feature] = lag_feature
                lag_features.append(sensor_feature)
                    
        # Set flag to True to avoid adding duplications
        feaures_added = True
    else:
        print('Features have already been added!')

    # Drop rows with NaN values
    dataset_transformed = dataset_transformed.dropna()

    # Remove the 'Unnamed: 0' column if it exists
    if "machine_status" in dataset_transformed.columns:
        dataset_transformed.drop(columns=["machine_status"], inplace=True)

    # Generate Train & Test splits based on assigned groups
    input_df = dataset_transformed[
        ~dataset_transformed['anomaly_group'].isin([4])
    ]

    test_df = dataset_transformed[
        dataset_transformed['anomaly_group'].isin([4])
    ].reset_index(drop=True)


    # Drop unnecessary columns
    input_df.drop(columns=['anomaly_group'], inplace=True)
    test_df.drop(columns=['anomaly_group'], inplace=True)
    # input_df.drop(columns=['anomaly_group'], inplace=True)

    # Define target, covariates, and time columns
    target_column = 'target'
    time_column_name = 'timestamp'
    time_features = [
        'timestamp_day_of_year', 
        'timestamp_month', 
        'timestamp_hour'
    ]
    covariate_cols = [
        *time_features, 
        *sensors, 
        *lag_features
    ]

    print("INPUT DF: ", input_df.head(10))
    print("INPUT DF Type: ", type(input_df))
    has_nan_overall = input_df.isna().any()
    print("HAS NAN: ", has_nan_overall)

    # Convert DataFrame into Darts TimeSeries format
    missing_date = True
    fill_na = True
    freq = '5min'

    input_target_ts = TimeSeries.from_dataframe(input_df, value_cols=target_column, time_col=time_column_name, freq=freq)
    input_covariates_ts = TimeSeries.from_dataframe(input_df, value_cols=covariate_cols, time_col=time_column_name, freq=freq)

    train_frac = 0.8
    train_split_point = int(len(input_target_ts) * train_frac)
    print(f"Train split point: {train_split_point} (fraction: {train_split_point})")
    train_target_ts = input_target_ts[:train_split_point]
    train_covariates_ts = input_covariates_ts[:train_split_point]

    val_target_ts = input_target_ts[train_split_point:]
    val_covariates_ts = input_covariates_ts[train_split_point:]
    # val_target_ts = TimeSeries.from_dataframe(val_df, value_cols=target_column, time_col=time_column_name, fill_missing_dates=missing_date, freq='5min')
    # val_covariates_ts = TimeSeries.from_dataframe(val_df, value_cols=covariate_cols, time_col=time_column_name, fill_missing_dates=missing_date, freq='5min')

    test_target_ts = TimeSeries.from_dataframe(test_df, value_cols=target_column, time_col=time_column_name, freq=freq)
    test_covariates_ts = TimeSeries.from_dataframe(test_df, value_cols=covariate_cols, time_col=time_column_name, freq=freq)

    # Scale Covariates
    covariate_scaler = Scaler(StandardScaler())

    train_covariates_ts_scaled = covariate_scaler.fit_transform(train_covariates_ts)
    val_covariates_ts_scaled = covariate_scaler.transform(val_covariates_ts)
    test_covariates_ts_scaled = covariate_scaler.transform(test_covariates_ts)
    input_covariates_ts_scaled = covariate_scaler.transform(input_covariates_ts)

    # List of all TimeSeries objects that will be used in model.fit()
    series_to_check = {
        "train_target_ts": train_target_ts,
        "val_target_ts": val_target_ts,
        "train_covariates_ts_scaled": train_covariates_ts_scaled,
        "val_covariates_ts_scaled": val_covariates_ts_scaled
    }


    found_invalid_values = False
    for name, ts in series_to_check.items():
        # Convert to a pandas DataFrame to use isnull() and isinf()
        df_check = ts.to_dataframe()

        # Check for NaN values
        nan_count = df_check.isnull().sum().sum()
        
        # Check for infinity values
        inf_count = np.isinf(df_check).sum().sum()
        
        if nan_count > 0 or inf_count > 0:
            print(f"CRITICAL ERROR: Found invalid values in '{name}'!")
            print(f"  - NaN count: {nan_count}")
            print(f"  - Infinity count: {inf_count}")
            found_invalid_values = True

    if not found_invalid_values:
        print("SUCCESS: No NaN or infinity values found in the input data. The problem is likely numerical instability.")
    else:
        print("\nACTION: Please investigate the source of the NaN/inf values. A likely cause is a feature with zero variance being scaled.")
        # You would then exit or debug from here
        # For example, to find a zero-variance column in your original data:
        # print(train_covariates_ts.pd_dataframe().describe()) 
        # Look for columns where 'std' is 0.0
        exit() # Stop the script before it crashes in the training loop

    # Optuna Hyperparameter Tuning
    n_optuna_trials = 10
    print(f"\n--- Starting Optuna Hyperparameter Tuning ({n_optuna_trials} trials) ---")
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_optuna_trials, callbacks=[print_callback])