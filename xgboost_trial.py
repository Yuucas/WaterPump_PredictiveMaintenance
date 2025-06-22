import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score
from xgboost import XGBClassifier


if __name__ == "__main__":

    print("--- Step 1: Load and Prepare Data ---")

    # Load the dataset. Ensure the CSV file is in the correct path.
    try:
        df = pd.read_csv(r'Forecasting\WaterPump_PredictiveMaintenance\dataset\pump_sensor_imputed.csv')
    except FileNotFoundError:
        print("Error: 'pump_sensor_data.csv' not found.")
        exit()

    # Convert timestamp to datetime objects and sort the data
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    print("Data loaded and sorted.")
    print(f"Dataset shape: {df.shape}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # --- Step 2: Engineer the Predictive Target Variable ---
    print("\n--- Step 2: Engineering the Target Variable: 'failure_in_3_days' ---")

    # Define the 3-day prediction horizon
    horizon = pd.Timedelta(days=1)

    # Find the exact timestamps of all 'BROKEN' events
    failure_events = df[df['machine_status'] == 'BROKEN']['timestamp']

    # Initialize the new target column with 0
    target_col_name = f"failure_in_{horizon.days}d"
    df[target_col_name] = 0

    # For each failure event, mark the preceding 3-day window as 1
    for event_time in failure_events:
        warning_start_time = event_time - horizon
        # Mark all rows within this window
        window_mask = (df['timestamp'] >= warning_start_time) & (df['timestamp'] < event_time)
        df.loc[window_mask, target_col_name] = 1

    print(f"Number of {target_col_name} (positive class) samples: {df[target_col_name].sum()}")
    print(f"Total samples in warning windows: {df[target_col_name].sum()} minutes of data.")

    # --- Step 3: Clean the Data for Modeling ---
    print("\n--- Step 3: Cleaning Data to Prevent Leakage ---")

    # We must remove the data during and after a failure to prevent the model from
    # learning from the failure event itself.
    # We keep only the 'NORMAL' operational data and the 'warning window' data.
    original_rows = df.shape[0]
    df_model = df[df['machine_status'] == 'NORMAL'].copy()
    cleaned_rows = df_model.shape[0]

    print(f"Removed {original_rows - cleaned_rows} rows corresponding to 'BROKEN' or 'RECOVERING' states.")

    # --- Step 4: Engineer Predictive Features ---
    print("\n--- Step 4: Engineering Rolling Window Features ---")

    df_model.set_index('timestamp', inplace=True)

    # For demonstration, we'll use a subset of sensors and window sizes.
    # In a real project, you would experiment with more sensors and windows.
    sensor_cols = [col for col in df.columns if 'sensor' in col]
    window_sizes = ['10min', '1h', '24h'] # 1 hour, 24 hours, 3 days

    # This loop can take a few minutes to run
    for window in window_sizes:
        print(f"  Calculating features for window size: {window}")
        for col in sensor_cols:
            # Calculate rolling statistics with corrected feature names
            df_model[f'{col}_mean_{window}'] = df_model[col].rolling(window, min_periods=1).mean()
            df_model[f'{col}_std_{window}'] = df_model[col].rolling(window, min_periods=1).std()
            df_model[f'{col}_skew_{window}'] = df_model[col].rolling(window, min_periods=1).skew()
            df_model[f'{col}_kurt_{window}'] = df_model[col].rolling(window, min_periods=1).kurt()


    print("De-fragmenting the DataFrame for better performance...")
    df_model = df_model.copy()

    # The first few rows will have NaN for std dev, so we backfill them
    df_model = df_model.fillna(method='bfill')

    # Also fill any remaining NaNs at the beginning with ffill
    df_model = df_model.fillna(method='ffill')

    print("Feature engineering complete.")
    print(f"New shape of the modeling dataframe: {df_model.shape}")

    # --- Step 5: Prepare Data for Modeling ---
    print("\n--- Step 5: Splitting Data for Training and Testing ---")

    # Define our features (X) and target (y)
    features = [col for col in df_model.columns if '_mean_' in col or '_std_' in col or '_skew_' in col or '_kurt_' in col]

    X = df_model[features]
    y = df_model[target_col_name]

    original_failure_times = df[df['machine_status'] == 'BROKEN']['timestamp']
    last_failure_time = original_failure_times.max()

    # The test set should start before the 3-day warning window of the last failure
    # We'll add a small buffer (e.g., 1 hour) to be safe.
    horizon = pd.Timedelta(days=3)
    buffer = pd.Timedelta(hours=1)
    split_timestamp = last_failure_time - horizon - buffer

    # The split is now based on this timestamp, not a percentage
    train_mask = X.index < split_timestamp
    test_mask = X.index >= split_timestamp

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"Training set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")
    print(f"Failures in training set: {y_train.sum()}")
    print(f"Failures in test set: {y_test.sum()}")

    # This check is crucial
    if y_test.sum() == 0:
        print("\nCRITICAL WARNING: The new split still resulted in a test set with no failures.")
        print("This may indicate an issue with the data or failure window logic. Please review.")
    else:
        print("\nSplit successful. The test set now contains failure warning data.")


    # Calculate scale_pos_weight to handle the massive class imbalance
    # This tells the model to pay much more attention to the rare positive class.
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = num_neg / num_pos
    scale_pos_weight = 15000

    print(f"Calculated scale_pos_weight for imbalance: {scale_pos_weight:.2f}")

    # Initialize and train the XGBoost model
    model = XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        n_estimators=750,      # Number of trees
        max_depth=10,           # Maximum depth of a tree
        learning_rate=0.01,
        gamma=0.5,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Model training complete.")

    # Save the trained XGBoost model
    model_filename = 'pump_failure_predictor_xgboost.json'
    model.save_model(model_filename)
    print(f"Model saved successfully to: {model_filename}")

    # Save the list of features the model was trained on
    features_filename = 'feature_list_xgboost.joblib'
    joblib.dump(features, features_filename)
    print(f"Feature list saved successfully to: {features_filename}")

    # --- Step 7: Evaluate Model Performance ---
    print("\n--- Step 7: Evaluating Model Performance on the Test Set ---")

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Probabilities for the positive class

    # Generate the classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Failure', 'Failure in 3 Days']))

    # Generate and plot the confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Plotting the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted No Failure', 'Predicted Failure'],
                yticklabels=['Actual No Failure', 'Actual Failure'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

    print("\n--- Final Interpretation ---")
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    print(f"Recall (Sensitivity): {recall:.2f}")
    print(f"Precision: {precision:.2f}")
    print("\nRecall is the most important metric here. It answers: 'Of all the actual 3-day failure windows, what percentage did we successfully predict?'")
    print("Precision tells us: 'Of all the failure alarms we raised, what percentage were correct?'")


