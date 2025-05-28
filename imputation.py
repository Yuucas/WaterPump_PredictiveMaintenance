import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Data processing
from sklearn.impute import IterativeImputer



if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    DATA_DIR = os.getenv("DATASET")
    IMP_DATASET_DIR = os.getenv("IMP_DATASET")

    # Load the dataset
    dataset = pd.read_csv(DATA_DIR)

    print("Initial dataset shape:", dataset.shape)
    print("Initial dataset columns:", dataset.columns)

    # Drop unnecessary columns
    dataset.drop(columns=["Unnamed: 0", "sensor_15"], inplace=True)

    column_name = "sensor_00"
    print(f"--- Statistics for {column_name} BEFORE imputation (on non-NaNs) ---")
    print(dataset[column_name].dropna().describe())

    # Store original columns that are not imputed, for later re-addition
    timestamp_col = dataset['timestamp']
    machine_status_col = dataset['machine_status']

    # # Handle missing values
    imp = IterativeImputer(max_iter=20, min_value=0, random_state=42)
    dataset_wo_time = dataset.drop(columns=["timestamp", "machine_status"])
    imputation_column_names = dataset_wo_time.columns
    dataset_wo_time = imp.fit_transform(dataset_wo_time)

    print("Check for missing values:", np.isnan(dataset_wo_time).sum())
    df_imputed = pd.DataFrame(dataset_wo_time, columns=imputation_column_names)
    print("Dataset after imputation:", df_imputed.head())

    print(f"\n--- Statistics for {column_name} AFTER imputation ---")
    print(df_imputed[column_name].describe())

    # Add back the desired columns
    df_imputed['timestamp'] = timestamp_col
    df_imputed['machine_status'] = machine_status_col

    # saving the dataframe
    df_imputed.to_csv(IMP_DATASET_DIR)