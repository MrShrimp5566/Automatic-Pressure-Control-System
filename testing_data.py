import pandas as pd
import numpy as np

def extract_features_from_window(window_data, setpoint):
    """
    Calculates key statistical features from a window of pressure data.
    
    Args:
        window_data (pd.Series): A pandas Series containing pressure readings.
        setpoint (float): The target pressure setpoint for the run.
    Returns:
        dict: A dictionary containing the calculated features.
    """
    if window_data.empty:
        return {
            'std_dev_pressure': 0,
            'mae_from_setpoint': 0,
            'zero_crossing_rate': 0
        }
    
    # Standard Deviation: Measure the amount of oscillation/instability.
    std_dev = window_data.std()

    # Mean Absolute Error: Measure how far, on average, the pressure is from the target.
    mae = (window_data - setpoint).abs().mean()

    # Zero-Crossing Rate: Counts how many times the signal coresses the setpoint.
    #   This is a great indicator of high-frequency oscillation.
    crossings = np.sum(np.diff(np.sign(window_data - setpoint)) != 0)
    
    return {
        'std_dev_pressure': std_dev,
        'mae_from_setpoint': mae,
        'zero_crossing_rate': crossings
    }

def simulate_real_time_processing(full_dataset, run_id_to_simulate):
    """
    Simulates the real-time processing of a single test run from the dataset.
    """
    run_data = full_dataset[full_dataset['Run ID'] == run_id_to_simulate].copy()
    setpoint = run_data.iloc[0]['Pressure (PSI)'] + 25 # Reconstruct the original setpoint

    # --- Simulate Parameters ---
    WINDOW_SIZE = 15 # seconds

    # This list will store the features calculated at each time step
    calculated_features_log = []

    # --- The Simulation Loop ---
    # We iterate through the data as if it were arriving in real-time
    for i in range(len(run_data)):
        # Define the current window of data
        start_index = max(0, i - WINDOW_SIZE + 1)
        current_window = run_data['Pressure (PSI)'].iloc[start_index : i + 1]

        # Extract features from the current window
        features = extract_features_from_window(current_window, setpoint)

        # Add the current time to the features dictionary for context
        features['Time (s)'] = run_data['Time (s)'].iloc[i]

        calculated_features_log.append(features)
    
    return pd.DataFrame(calculated_features_log)

if __name__ == "__main__":
    try:
        # Load the dataset we created in the previous step
        dataset = pd.read_csv('pid_tuning_experimental_data.csv')
    except FileNotFoundError:
        print("Error: 'pid_tuning_experimental_data.csv' not found.")
        print("Please run the previous script to generate the dataset first.")
        exit()

    # --- Simulate and analyze the "Before Tuning" run ---
    print("--- Simulating Run 3 ('Before Tuning') ---")
    features_before_tuning = simulate_real_time_processing(dataset, run_id_to_simulate=3)

    # Display the average feature values during the unstable recovery period
    recovery_period_before = features_before_tuning[(features_before_tuning['Time (s)'] > 20) & (features_before_tuning['Time (s)'] <= 100)]
    print("\nAverage Features during Unstable Period (Before Tuning):")
    print(recovery_period_before[['std_dev_pressure', 'mae_from_setpoint', 'zero_crossing_rate']].mean())

    # --- Simulate and analyze the "After Tuning" run ---
    print("\n" + "="*50)
    print("--- Simulating Run 4 ('After Tuning') ---")
    features_after_tuning = simulate_real_time_processing(dataset, run_id_to_simulate=4)
    
    # Display the average feature values during the stable recovery period
    recovery_period_after = features_after_tuning[(features_after_tuning['Time (s)'] > 20) & (features_after_tuning['Time (s)'] < 100)]
    print("\nAverage Features during Stable Period (After Tuning):")
    print(recovery_period_after[['std_dev_pressure', 'mae_from_setpoint', 'zero_crossing_rate']].mean())
    print("="*50) 