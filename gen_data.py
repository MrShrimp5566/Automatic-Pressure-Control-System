import pandas as pd
import numpy as np
import os

def generate_matched_data(samples_per_state=1000):
    """
    Generates a synthetic dataset simulating a pressure valve system with three states: NORMAL, HIGH PRESSURE, and LOW PRESSURE.
    Each state has distinct characteristics for pressure, temperature, and flow rate.
    Args:
        samples_per_state (int): The number of data points to generate for each state.

    Returns:
        pandas.DataFrame: A shuffled DataFrame with the complete dataset.
    """
    all_states_df = []

    # --- NORMAL State ---
    pressure_normal = 50 + 8 * np.sin(np.linspace(0, 50, samples_per_state)) + np.random.normal(0, 2, samples_per_state)
    temp_normal = 28 + np.random.normal(0, 1.5, samples_per_state)
    flow_normal = 12 + np.random.normal(0, 2, samples_per_state)
    df_normal = pd.DataFrame({
        'Pressure (PSI)': pressure_normal, 'Temperature (°C)': temp_normal,
        'Flow Rate (L/min)': flow_normal, 'Valve State': 'HOLD', 'Alert Status': 'NORMAL'
    })
    all_states_df.append(df_normal)

    # --- HIGH PRESSURE State ---
    pressure_high = 65 + np.random.normal(0, 3, samples_per_state)
    temp_high = 29.5 + np.random.normal(0, 1.5, samples_per_state)
    flow_high = 14 + np.random.normal(0, 2, samples_per_state)
    df_high = pd.DataFrame({
        'Pressure (PSI)': pressure_high, 'Temperature (°C)': temp_high,
        'Flow Rate (L/min)': flow_high, 'Valve State': 'OPEN', 'Alert Status': 'HIGH PRESSURE'
    })
    all_states_df.append(df_high)
    
    # --- LOW PRESSURE State ---
    pressure_low = 35 + np.random.normal(0, 3, samples_per_state)
    temp_low = 26.5 + np.random.normal(0, 1.5, samples_per_state)
    flow_low = 10 + np.random.normal(0, 2, samples_per_state)
    df_low = pd.DataFrame({
        'Pressure (PSI)': pressure_low, 'Temperature (°C)': temp_low,
        'Flow Rate (L/min)': flow_low, 'Valve State': 'CLOSE', 'Alert Status': 'LOW PRESSURE'
    })
    all_states_df.append(df_low)

    # --- Combine and Add Time ---
    combined_df = pd.concat(all_states_df, ignore_index=True)
    shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)
    shuffled_df.insert(0, 'Time (s)', shuffled_df.index * 5)
    
    return shuffled_df

# ==============================================================================
# SCRIPT EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # --- Generate a dataset and save it ---
    synthetic_dataset = generate_matched_data(samples_per_state=1000)
    
    file_path = 'synthetic_pressure_valve_data.csv'
    
    write_header = not os.path.exists(file_path)

    synthetic_dataset.to_csv(file_path, mode='a', header=write_header, index=False)
    
    # A single print statement to confirm completion
    if write_header:
        print(f"New file created with {len(synthetic_dataset)} samples at '{file_path}'.")
    else:
        print(f"{len(synthetic_dataset)} samples appended to '{file_path}'.")