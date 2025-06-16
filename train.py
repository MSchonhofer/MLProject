import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

# Load data
print("[INFO] Loading training data...")
X = np.load('data/x_data_corrected.npy')    # shape: (num_voxels, 44)
y = np.load('data/y_labels.npy')            # shape: (num_voxels, )

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train/val
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Simple MLP
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
print("[INFO] Training model...")
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=256,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
)

# Save model and scaler
os.makedirs('models', exist_ok=True)
model.save('models/voxel_model.h5')
joblib.dump(scaler, 'models/scaler.pkl')

print("[INFO] Training complete. Model saved to models/voxel_model.h5")


