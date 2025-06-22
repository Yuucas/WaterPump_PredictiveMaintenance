import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np

#
if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    IMP_DATASET = os.getenv("IMP_DATASET")
    RESAMPLED_DATASET_DIR = os.getenv("RESAMPLED_DATASET")

    # Load the dataset
    dataset = pd.read_csv(IMP_DATASET)

    # Drop unnecessary columns
    dataset.drop(columns=["Unnamed: 0"], inplace=True)

    status_counts = dataset['machine_status'].value_counts()
    print("Counts of each unique string in 'machine_status':")
    print(status_counts)

    # Convert timestamp and sort
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp'])
    dataset.set_index('timestamp', inplace=True)

    # --- 2. Define Custom Aggregation Function for 'machine_status' ---
    def custom_status_agg(status_series):
        """
        Aggregates machine status for a time window.
        Prioritizes 'BROKEN'. If 'BROKEN' is present, the window is 'BROKEN'.
        Otherwise, falls back to the mode of other statuses.
        """
        unique_statuses_in_window = status_series.unique()
        if 'BROKEN' in unique_statuses_in_window:
            return 'BROKEN'
        else:
            # Fallback to mode if 'BROKEN' is not present
            modes = status_series.mode()
            if not modes.empty:
                return modes[0] # Return the first mode if multiple exist
            else:
                return 0 # Or some other default if the series was empty/all NaN

    # --- Define Aggregation Logic ---
    agg_functions = {}
    for column in dataset.columns:
        if pd.api.types.is_numeric_dtype(dataset[column]):
            agg_functions[column] = 'mean'
        elif column == 'machine_status':
            agg_functions[column] = custom_status_agg
        else:
            agg_functions[column] = 'first' 

    # --- 3. Resample to 5-minute Frequency ---
    dataset_5min_freq = dataset.resample('5min').agg(agg_functions)
    print(dataset_5min_freq.head(10))

    # dataset_5min_freq = dataset_5min_freq.fillna(method='ffill').fillna(method='bfill')
    # print("Filling NaNs created during resampling...")
    df_resampled = dataset_5min_freq.fillna(method='ffill')
    # Add a backfill just in case there are NaNs at the very beginning
    df_resampled_v2 = df_resampled.fillna(method='bfill')

    status_counts = df_resampled_v2['machine_status'].value_counts()
    print("Counts of each unique string in 'machine_status':")
    print("5Min: ", status_counts)

    # saving the dataframe
    df_resampled_v2.to_csv(RESAMPLED_DATASET_DIR)