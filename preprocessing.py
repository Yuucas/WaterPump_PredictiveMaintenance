import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Data processing
from feature_engine.selection import RecursiveFeatureElimination
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from feature_engine.creation import DecisionTreeFeatures
from feature_engine.datetime import DatetimeFeatures
from feature_engine.imputation import MeanMedianImputer

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def apply_datetime_features(dataframe, variable_name="Date"):

    dtfs = DatetimeFeatures(
        variables=variable_name,
        features_to_extract=["day_of_year", "month", "hour"],
        drop_original=False
    )

    dfs_transformed = dtfs.fit_transform(dataframe)

    return dfs_transformed



if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    RESAMPLED_DATASET_DIR = os.getenv("RESAMPLED_DATASET")
    OUTPUT_DIR = os.getenv("OUTPUT_DIR_v2")

    # Load the dataset
    dataset = pd.read_csv(RESAMPLED_DATASET_DIR)

    print("Initial dataset shape:", dataset.shape)
    print("Initial dataset columns:", dataset.columns)
    print("Initial dataset dtypes:", dataset.describe(include="all"))

    # Apply datetime features
    dataset_transformed = apply_datetime_features(dataset, variable_name="timestamp")

    # Drop unnecessary columns
    time_stamp = dataset_transformed["timestamp"]
    dataset_transformed.drop(columns=["timestamp"], inplace=True)

    # Encode categorical variables
    label_encoder = LabelEncoder()
    for column in dataset_transformed.select_dtypes(include=['object']).columns:
        dataset_transformed[column] = label_encoder.fit_transform(dataset_transformed[column])

    # Feature selection using Recursive Feature Elimination
    X = dataset_transformed.drop(columns=["machine_status"], axis=1)
    y = dataset_transformed["machine_status"]

    # Check if any NaN values exist in the entire array
    has_nans = np.isnan(X).any()
    print(f"X has any NaNs: {has_nans}")

    print(X, type(X))
 
    # --- Feature Elimination ---
    # initialize linear regresion estimator
    rfr_model = RandomForestRegressor(n_estimators=100, 
                                      criterion='squared_error', 
                                      max_features='sqrt', 
                                      )
    
    rfe = RecursiveFeatureElimination(estimator=rfr_model, scoring="r2", cv=3)
    processed_df = rfe.fit_transform(X, y)
    
    # Save the processed data
    processed_df["machine_status"] = y
    processed_df["timestamp"] = time_stamp
    processed_df.to_csv(OUTPUT_DIR, index=False)