import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Paths
X_TEST_PATH = 'data/test_patient/x_test.npy'
MODEL_PATH = 'models/voxel_model.h5'
SCALER_PATH = 'models/scaler.pkl'
OUTPUT_PATH = 'data/test_patient/probability_map.npy'

#Load x_test features
print("[INFO] Loading test patient features...")
X_test = np.load(X_TEST_PATH)  # shape: (num_voxels, 44)

# Load model and scaler
print("[INFO] Loading model and scaler...")
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Normalize features
X_scaled = scaler.transform(X_test)

# Predict cancer probability for each voxel
print("[INFO] Predicting cancer probabilities...")
predictions = model.predict(X_scaled).flatten()  # shape: (num_voxels,)

# Save output
np.save(OUTPUT_PATH, predictions)
print(f"[INFO] Saved voxel-level probability map to {OUTPUT_PATH}")



