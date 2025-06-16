import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from data_utils import load_data, isolate_and_normalize
from unet_model import build_unet_model
from loss_functions import dice_coefficient

BASE_FILTERS = 32
INPUT_SHAPE = (308, 384, 1)
MODEL_PATH = f"./models/seg_model_N{BASE_FILTERS}.weights.h5"

model = build_unet_model(base_filters=BASE_FILTERS, input_shape=INPUT_SHAPE)
model.load_weights(MODEL_PATH)

t2_vols, cap_vols = load_data()
t2_vols, cap_vols = t2_vols.transpose(0, 3, 2, 1), cap_vols.transpose(0, 3, 2, 1)
X, y = isolate_and_normalize(t2_vols, cap_vols)

y_pred = model.predict(X, batch_size=4)

threshold = 0.5
binary_preds = (y_pred > threshold).astype(np.uint8)
intersection = np.sum(binary_preds * y)
union = np.sum(binary_preds) + np.sum(y)
iou = intersection / (union - intersection + 1e-6)
dice = dice_coefficient(y, y_pred)

print(f"IoU: {iou:.4f}")
print(f"Dice Coefficient: {dice:.4f}")

errors = np.where(np.abs(binary_preds - y) > 0.5)[0][:6]

plt.figure(figsize=(12, 6))
for idx, i in enumerate(errors):
    plt.subplot(2, 6, idx+1)
    plt.imshow(X[i].squeeze(), cmap='gray')
    plt.title("Input")
    plt.axis('off')
    plt.subplot(2, 6, idx+7)
    plt.imshow(X[i].squeeze(), cmap='gray')
    plt.imshow(binary_preds[i].squeeze(), alpha=0.5, cmap='Reds')
    plt.title("Prediction")
    plt.axis('off')
plt.tight_layout()
plt.show()
