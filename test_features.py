import os
import numpy as np
import nibabel as nib

TEST_DIR = 'data/test_patient/'
OUT_PATH = os.path.join(TEST_DIR, 'x_test.npy')
MASK_SAVE_PATH = os.path.join(TEST_DIR, 'mask_indices.npy')

# Load modalities
def load_test_modalities(folder):
    t2 = adc = dwi = dce = mask_prost = None
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if 't2' in fname:
            t2 = nib.load(fpath).get_fdata()
        elif 'adc' in fname and 'resampled' in fname:
            adc = nib.load(fpath).get_fdata()
        elif 'epi' in fname and 'resampled' in fname:
            dwi = nib.load(fpath).get_fdata()
        elif 't1' in fname and 'resampled' in fname:
            dce = nib.load(fpath).get_fdata()
        elif 'prostate_mask' in fname:
            mask_prost = nib.load(fpath).get_fdata()
    return t2, adc, dwi, dce, mask_prost

t2, adc, dwi, dce, mask = load_test_modalities(TEST_DIR)
assert all(m is not None for m in [t2, adc, dwi, dce, mask]), "Missing one or more MRI sequences!"

# Extract features from prostate voxels only
mask = mask > 0  # ensure boolean
mask_indices = np.argwhere(mask)

# Feature extraction
t2_feat = t2[mask].reshape(-1, 1)
adc_feat = adc[mask].reshape(-1, 1)

dwi_feat = np.stack([dwi[..., i][mask].reshape(-1) for i in range(dwi.shape[-1])], axis=-1)
dce_feat = np.stack([dce[..., i][mask].reshape(-1) for i in range(dce.shape[-1])], axis=-1)

print("Shapes:")
print("T2: ", t2_feat.shape)
print("ADC:", adc_feat.shape)
print("DWI:", dwi_feat.shape)
print("DCE:", dce_feat.shape)

# Combine into final (N, 44) feature vector
X_test = np.concatenate([t2_feat, adc_feat, dwi_feat, dce_feat], axis=1)

# Save output
np.save(OUT_PATH, X_test)
np.save(MASK_SAVE_PATH, mask_indices)
print(f"[INFO] Saved test features to {OUT_PATH} with shape {X_test.shape}")
print(f"[INFO] Saved mask indices to {MASK_SAVE_PATH}")

