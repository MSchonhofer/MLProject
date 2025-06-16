import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from data_utils import load_data, isolate_and_normalize
from unet_model import build_unet_model

BASE_FILTERS = 32
INPUT_SHAPE = (308, 384, 1)
MODEL_PATH = f"./models/seg_model_N{BASE_FILTERS}.weights.h5"
THRESHOLD = 0.5

t2_vols, cap_vols = load_data()
t2_vols, cap_vols = t2_vols.transpose(0, 3, 2, 1), cap_vols.transpose(0, 3, 2, 1)
X, y = isolate_and_normalize(t2_vols, cap_vols)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = build_unet_model(base_filters=BASE_FILTERS, input_shape=INPUT_SHAPE)
model.load_weights(MODEL_PATH)

y_pred = model.predict(X_test, batch_size=4)
binary_preds = (y_pred > THRESHOLD).astype(np.uint8)

intersection = np.sum(binary_preds * y_test)
union = np.sum(binary_preds) + np.sum(y_test)
iou = intersection / (union - intersection + 1e-6)
dice = (2. * intersection) / (np.sum(binary_preds) + np.sum(y_test) + 1e-6)

print(f"IoU: {iou:.4f}")
print(f"Dice Coefficient: {dice:.4f}")

errors = np.where(np.abs(binary_preds - y_test) > 0.5)[0][:6]

plt.figure(figsize=(12, 9))
for idx, i in enumerate(errors):
    plt.subplot(3, 6, idx + 1)
    plt.imshow(X_test[i].squeeze(), cmap='gray')
    plt.title("Input")
    plt.axis('off')

    plt.subplot(3, 6, idx + 7)
    plt.imshow(X_test[i].squeeze(), cmap='gray')
    plt.imshow(binary_preds[i].squeeze(), alpha=0.5, cmap='Reds')
    plt.title("Prediction")
    plt.axis('off')

    plt.subplot(3, 6, idx + 13)
    plt.imshow(y_test[i].squeeze(), cmap='gray')
    plt.title("Ground Truth")
    plt.axis('off')

plt.tight_layout()
plt.show()

