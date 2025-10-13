import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# PART 1: PREPARE THE DATA (Same logic as before)
# ==============================================================================

def extract_features_from_window(window_data, setpoint):
    if len(window_data) < 2:
        return {'std_dev_pressure': 0, 'mae_from_setpoint': 0, 'zero_crossing_rate': 0}
    std_dev = window_data.std()
    mae = (window_data - setpoint).abs().mean()
    crossings = np.sum(np.diff(np.sign(window_data - setpoint)) != 0)
    return {'std_dev_pressure': std_dev, 'mae_from_setpoint': mae, 'zero_crossing_rate': crossings}

def create_feature_dataset(df):
    WINDOW_SIZE = 15
    all_features = []
    for run_id in df['Run ID'].unique():
        run_data = df[df['Run ID'] == run_id].copy()
        setpoint = np.mean(run_data['Pressure (PSI)'].iloc[250:300])
        label = 1 if run_data.iloc[0]['Tuning State'] == 'Before Tuning' else 0
        for i in range(WINDOW_SIZE, len(run_data)):
            window = run_data['Pressure (PSI)'].iloc[i - WINDOW_SIZE : i]
            features = extract_features_from_window(window, setpoint)
            features['label'] = label
            all_features.append(features)
    return pd.DataFrame(all_features)

try:
    raw_df = pd.read_csv('pid_tuning_experimental_data.csv')
except FileNotFoundError:
    print("Error: 'pid_tuning_experimental_data.csv' not found.")
    exit()

feature_df = create_feature_dataset(raw_df)
X = feature_df[['std_dev_pressure', 'mae_from_setpoint', 'zero_crossing_rate']]
y = feature_df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================================================================
# PART 2: BUILD AND TRAIN THE MODEL
# ==============================================================================

# Using keras directly since we imported it from tensorflow
model = keras.Sequential([
    keras.layers.Input(shape=(3,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("--- Starting Model Training ---")
model.fit(X_train, y_train, epochs=10, validation_split=0.2, verbose=1)

print("\n--- Evaluating Model on Unseen Test Data ---")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")
print("="*60)

# ==============================================================================
# PART 3: CONVERT MODEL FOR TINYML DEPLOYMENT
# ==============================================================================

print("\n--- Converting Model to TensorFlow Lite format ---")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Define the representative dataset generator
def representative_data_gen():
    for i in range(100):
        # Explicitly cast to float32, which is required
        yield [X_train[i].astype(np.float32).reshape(1, 3)]

converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model_quant = converter.convert()

tflite_model_path = 'pid_model.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model_quant)

print(f"\nSuccessfully converted and saved the quantized model to '{tflite_model_path}'")
model_size = os.path.getsize(tflite_model_path)
print(f"Final Model Size: {model_size} bytes")
print("="*60)
print("\nTo convert the model to a C header file, run the following command in your terminal:")
print(f"xxd -i {tflite_model_path} > model.h")