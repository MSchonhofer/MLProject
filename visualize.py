import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os

TEST_DIR = 'data/test_patient/'
t2_path = [f for f in os.listdir(TEST_DIR) if 't2' in f][0]
mask_path = [f for f in os.listdir(TEST_DIR) if 'prostate_mask' in f][0]

# Load data
t2_img = nib.load(os.path.join(TEST_DIR, t2_path)).get_fdata()
mask = nib.load(os.path.join(TEST_DIR, mask_path)).get_fdata() > 0
prob_flat = np.load(os.path.join(TEST_DIR, 'probability_map.npy'))
indices = np.load(os.path.join(TEST_DIR, 'mask_indices.npy'))

# Reconstruct 3D probability map
prob_map = np.zeros(t2_img.shape)
for i, (x, y, z) in enumerate(indices):
    prob_map[x, y, z] = prob_flat[i]

# Output directory for overlays
OUTPUT_DIR = os.path.join(TEST_DIR, 'slices')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Loop through slices that contain prostate
z_slices = np.unique(indices[:, 2])
print(f"[INFO] Generating overlays for {len(z_slices)} slices...")

for slice_idx in z_slices:
    slice_idx = int(slice_idx)

    plt.figure(figsize=(10, 6))
    plt.imshow(t2_img[:, :, slice_idx], cmap='gray')
    plt.imshow(prob_map[:, :, slice_idx], cmap='hot', alpha=0.5)
    plt.title(f"Predicted Cancer Probability Map (slice {slice_idx})")
    plt.axis('off')
    plt.colorbar(label='Probability')

    save_path = os.path.join(OUTPUT_DIR, f"slice_{slice_idx:03}.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

print(f"[INFO] Overlays saved to: {OUTPUT_DIR}")