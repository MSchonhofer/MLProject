import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from data_utils import load_data, isolate_and_normalize
from unet_model import build_unet_model
from loss_functions import bce_dice_loss  # NEW import

# Constants
BASE_FILTERS = 32
INPUT_SHAPE = (308, 384, 1)
MODEL_PATH = f"./models/seg_model_N{BASE_FILTERS}.weights.h5"
THRESHOLD = 0.2  # Lower threshold to pick up low-confidence lesions

# Load and preprocess data
t2_vols, cap_vols = load_data()
t2_vols, cap_vols = t2_vols.transpose(0, 3, 2, 1), cap_vols.transpose(0, 3, 2, 1)
X, y = isolate_and_normalize(t2_vols, cap_vols)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Load model
model = build_unet_model(base_filters=BASE_FILTERS, input_shape=INPUT_SHAPE)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=bce_dice_loss)
model.load_weights(MODEL_PATH)

# Predict
y_pred = model.predict(X_test, batch_size=4)
binary_preds = (y_pred > THRESHOLD).astype(np.uint8)

# Metrics
intersection = np.sum(binary_preds * y_test)
union = np.sum(binary_preds) + np.sum(y_test)
iou = intersection / (union - intersection + 1e-6)
dice = (2. * intersection) / (np.sum(binary_preds) + np.sum(y_test) + 1e-6)

print(f"IoU: {iou:.4f}")
print(f"Dice Coefficient: {dice:.4f}")

# Visualize predictions, ground truth, and heatmaps
plt.figure(figsize=(12, 12))
for idx in range(6):
    plt.subplot(4, 6, idx + 1)
    plt.imshow(X_test[idx].squeeze(), cmap='gray')
    plt.title("Input")
    plt.axis('off')

    plt.subplot(4, 6, idx + 7)
    plt.imshow(X_test[idx].squeeze(), cmap='gray')
    plt.imshow(binary_preds[idx].squeeze(), alpha=0.5, cmap='Reds')
    plt.title("Prediction")
    plt.axis('off')

    plt.subplot(4, 6, idx + 13)
    plt.imshow(y_test[idx].squeeze(), cmap='gray')
    plt.title("Ground Truth")
    plt.axis('off')

    plt.subplot(4, 6, idx + 19)
    plt.imshow(y_pred[idx].squeeze(), cmap='hot')
    plt.title("Raw Prob Map")
    plt.axis('off')

plt.tight_layout()
plt.show()

