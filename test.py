import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data_utils import load_test_patient
from unet_model import build_unet_model
from loss_functions import dice_coefficient
from visualize import compare_predictions

BASE_FILTERS = 32
INPUT_SHAPE = (308, 384, 1)
MODEL_PATH = f"./models/seg_model_N{BASE_FILTERS}.weights.h5"

# Load model
model = build_unet_model(base_filters=BASE_FILTERS, input_shape=INPUT_SHAPE)
model.load_weights(MODEL_PATH)

# Load test patient data
X_test, y_test = load_test_patient('./training_data/test_patient')
print("Test patient shape:", X_test.shape)

# Predict
y_pred = model.predict(X_test, batch_size=4)

# Evaluate
threshold = 0.2
binary_preds = (y_pred > threshold).astype(np.uint8)
intersection = np.sum(binary_preds * y_test)
union = np.sum(binary_preds) + np.sum(y_test)
iou = intersection / (union - intersection + 1e-6)
dice = dice_coefficient(y_test, y_pred)

print(f"IoU: {iou:.4f}")
print(f"Dice Coefficient: {dice:.4f}")

compare_predictions(X_test, y_test, y_pred, threshold=threshold)

import matplotlib.pyplot as plt

# Flatten all predicted voxel probabilities
all_preds = y_pred.flatten()

# Plot histogram of probabilities
plt.figure(figsize=(8, 4))
plt.hist(all_preds, bins=50, color='skyblue', edgecolor='black')
plt.title("Histogram of Predicted Voxel Probabilities")
plt.xlabel("Predicted Probability")
plt.ylabel("Voxel Count")
plt.grid(True)
plt.tight_layout()
plt.show()


