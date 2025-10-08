import pandas as pd
import numpy as np
import os

def generate_pid_test_run(run_id, setpoint, disturbance, tuning_state):
    """
    Generates a single test run simulating a PID controller's response.
    The response characteristics are determined by the 'tuning_state'.

    Args:
        run_id (int): Identifier for this test run.
        setpoint (float): The target pressure for the controller.
        disturbance (float): The magnitude of the pressure drop.
        tuning_state (str): 'Before Tuning' or 'After Tuning'.

    Returns:
        pandas.DataFrame: A DataFrame containing the data for one test run.
    """
    duration = 300  # seconds
    sample_rate = 1 # samples per second
    num_samples = duration * sample_rate
    time = np.linspace(0, duration, num_samples)
    
    # --- Define PID performance parameters based on tuning state ---
    if tuning_state == 'Before Tuning':
        # Poorly tuned: slow decay (oscillates for a long time), higher frequency
        decay = 0.010
        frequency = 0.12
    else:  # 'After Tuning'
        # Well-tuned: faster decay (stabilizes quickly), lower frequency
        decay = 0.040
        frequency = 0.08
        
    # --- Simulate the PID Response using a Damped Sine Wave ---
    recovery_phase = time[time > 10] - 10
    damped_wave = disturbance * np.exp(-decay * recovery_phase) * np.cos(frequency * recovery_phase)
    pressure = np.full_like(time, setpoint)
    pressure[time <= 10] += disturbance
    pressure[time > 10] += damped_wave
    pressure += np.random.normal(0, 0.25, num_samples)
    
    # --- Simulate other related parameters ---
    valve_open_pct = 50 + -2 * (np.gradient(pressure, time))
    valve_open_pct = np.clip(valve_open_pct, 10, 90)
    flow_rate = valve_open_pct * 0.18 + np.random.normal(0, 0.5, num_samples)
    temperature = 25 + (pressure / setpoint) * 5 + np.random.normal(0, 0.5, num_samples)
    
    # --- Assemble the DataFrame ---
    df = pd.DataFrame({
        'Run ID': run_id,
        'Tuning State': tuning_state,
        'Time (s)': time,
        'Pressure (PSI)': pressure,
        'Valve Open (%)': valve_open_pct,
        'Flow Rate (L/min)': flow_rate,
        'Temperature (°C)': temperature
    })
    
    return df

# ==============================================================================
# SCRIPT EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # --- Define new test runs to be generated for this execution ---
    # This list now includes the 'Tuning State' for each run
    test_run_definitions = [
        {'run_id': 1, 'setpoint': 65, 'disturbance': -20, 'tuning': 'Before Tuning'},
        {'run_id': 2, 'setpoint': 65, 'disturbance': -20, 'tuning': 'After Tuning'},
        {'run_id': 3, 'setpoint': 70, 'disturbance': -25, 'tuning': 'Before Tuning'},
        {'run_id': 4, 'setpoint': 70, 'disturbance': -25, 'tuning': 'After Tuning'},
    ]
    
    all_runs_df = []
    for definition in test_run_definitions:
        run_df = generate_pid_test_run(
            run_id=definition['run_id'],
            setpoint=definition['setpoint'],
            disturbance=definition['disturbance'],
            tuning_state=definition['tuning']
        )
        all_runs_df.append(run_df)
    
    final_dataset = pd.concat(all_runs_df, ignore_index=True)
    
    file_path = 'pid_tuning_experimental_data.csv'
    
    # --- Logic to Append or Create New File ---
    write_header = not os.path.exists(file_path)
    final_dataset.to_csv(file_path, mode='a', header=write_header, index=False)