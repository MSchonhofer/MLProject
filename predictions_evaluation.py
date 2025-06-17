import numpy as np
import nibabel as nib
import os
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

TEST_DIR = 'data/test_patient/'
prob_map = np.load(os.path.join(TEST_DIR, 'probability_map.npy'))
indices = np.load(os.path.join(TEST_DIR, 'mask_indices.npy'))

# Reconstruct 3D prediction
shape = nib.load(os.path.join(TEST_DIR, 'prostate_mask.nii.gz')).get_fdata().shape
pred_full = np.zeros(shape)
for i, (x, y, z) in enumerate(indices):
    pred_full[x, y, z] = prob_map[i]

# Gound truth mask (cap_mask = tumor)
cap_mask = nib.load(os.path.join(TEST_DIR, 'cap_mask.nii.gz')).get_fdata() > 0
gt_labels = cap_mask[cap_mask > 0]  # all should be 1
pred_values = pred_full[cap_mask > 0]

# Evaluation over whole prostate
prostate_mask = nib.load(os.path.join(TEST_DIR, 'prostate_mask.nii.gz')).get_fdata() > 0
y_true = cap_mask[prostate_mask]     # binary labels in prostate
y_pred = pred_full[prostate_mask]    # model predictions in prostate

# Threshold predictions for binary evaluation
threshold = 0.75
y_pred_binary = (y_pred >= threshold).astype(int)

# Metrics
dice = 2 * np.sum((y_pred_binary & y_true)) / (np.sum(y_pred_binary) + np.sum(y_true) + 1e-8)
auc = roc_auc_score(y_true.flatten(), y_pred.flatten())
cm = confusion_matrix(y_true.flatten(), y_pred_binary.flatten())
precision = precision_score(y_true.flatten(), y_pred_binary.flatten())
recall = recall_score(y_true.flatten(), y_pred_binary.flatten())
f1 = f1_score(y_true.flatten(), y_pred_binary.flatten())

# Print results
print(f"\n[Evaluation Results]")
print(f"Dice Score:     {dice:.4f}")
print(f"ROC AUC:        {auc:.4f}")
print(f"Precision:      {precision:.4f}")
print(f"Recall:         {recall:.4f}")
print(f"F1 Score:       {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(y_pred.flatten(), bins=50, color='steelblue', edgecolor='black')
plt.title("Histogram of Predicted Cancer Probabilities (within prostate)")
plt.xlabel("Predicted Probability")
plt.ylabel("Number of Voxels")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(TEST_DIR, 'probability_histogram.png'))
plt.show()

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_true.flatten(), y_pred.flatten())
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve – Voxel-wise Cancer Prediction')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(TEST_DIR, 'roc_curve.png'))
plt.show()

# Function for finding best threshold
def compute_metrics(y_true, y_pred_binary):
    tp = np.sum((y_true == 1) & (y_pred_binary == 1))
    fp = np.sum((y_true == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true == 1) & (y_pred_binary == 0))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return dice, precision, recall, f1

# Sweep thresholds
thresholds = np.arange(0.1, 0.91, 0.05)
dice_list = []
prec_list = []
rec_list = []
f1_list = []

for t in thresholds:
    y_bin = (y_pred >= t).astype(int)
    d, p, r, f1 = compute_metrics(y_true, y_bin)
    dice_list.append(d)
    prec_list.append(p)
    rec_list.append(r)
    f1_list.append(f1)

plt.figure(figsize=(10, 6))
plt.plot(thresholds, dice_list, label='Dice Score', marker='o')
plt.plot(thresholds, f1_list, label='F1 Score', marker='s')
plt.plot(thresholds, prec_list, label='Precision', linestyle='--')
plt.plot(thresholds, rec_list, label='Recall', linestyle='--')
plt.xlabel("Threshold")
plt.ylabel("Metric Score")
plt.title("Threshold Sweep: Dice, F1, Precision, Recall")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(TEST_DIR, 'threshold_sweep_metrics.png'))
plt.show()

