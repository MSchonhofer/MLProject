import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os

TEST_DIR = 'data/test_patient/'
t2_path = [f for f in os.listdir(TEST_DIR) if 't2' in f][0]

# Load files
t2 = nib.load(os.path.join(TEST_DIR, t2_path)).get_fdata()
prostate_mask = nib.load(os.path.join(TEST_DIR, 'prostate_mask.nii.gz')).get_fdata() > 0
cap_mask = nib.load(os.path.join(TEST_DIR, 'cap_mask.nii.gz')).get_fdata() > 0
prob = np.load(os.path.join(TEST_DIR, 'probability_map.npy'))
indices = np.load(os.path.join(TEST_DIR, 'mask_indices.npy'))

# Rebuild 3D probability map
prob_map = np.zeros(t2.shape)
for i, (x, y, z) in enumerate(indices):
    prob_map[x, y, z] = prob[i]

output_dir = os.path.join(TEST_DIR, 'gt_comparison')
os.makedirs(output_dir, exist_ok=True)

# Loop through slices with tumor
z_slices = np.unique(np.argwhere(cap_mask)[:, 2])
print(f"[INFO] Visualizing {len(z_slices)} slices with tumor...")

for z in z_slices:
    z = int(z)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(t2[:, :, z], cmap='gray')

    # Overlay prediction heatmap
    im = ax.imshow(prob_map[:, :, z], cmap='hot', alpha=0.5)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label='Predicted Probability')

    # Overlay ground truth contour
    ax.contour(cap_mask[:, :, z], colors='lime', linewidths=1.5, linestyles='--', levels=[0.5])

    ax.set_title(f"Prediction vs. Ground Truth – Slice {z}")
    ax.axis('off')

    out_path = os.path.join(output_dir, f'slice_{z:03}.png')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

print(f"[INFO] Saved comparison images to {output_dir}")
