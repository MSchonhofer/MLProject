import os
import numpy as np
import nibabel as nib

def crop_uniform(images):
    rows, cols = zip(*[(img.shape[0], img.shape[1]) for img in images])
    min_row, min_col = min(rows), min(cols)

    for idx, img in enumerate(images):
        row_diff, col_diff = img.shape[0] - min_row, img.shape[1] - min_col
        row_slice = slice(row_diff // 2, -((row_diff + 1) // 2) or None)
        col_slice = slice(col_diff // 2, -((col_diff + 1) // 2) or None)
        images[idx] = img[row_slice, col_slice, ...]

def isolate_and_normalize(t2_imgs, mask_imgs):
    norm_t2, norm_mask = [], []

    for t2_stack, mask_stack in zip(t2_imgs, mask_imgs):
        for t2_slice, mask_slice in zip(t2_stack, mask_stack):
            if np.sum(mask_slice) > 0:
                norm_t2.append((t2_slice / np.max(t2_slice))[..., np.newaxis])
                norm_mask.append((mask_slice / 255.0)[..., np.newaxis])

    return np.array(norm_t2), np.array(norm_mask)

def load_data(dataset_path='./training_data/patients'):
    t2_data, mask_data = [], []

    for patient in sorted(os.listdir(dataset_path)):
        patient_path = os.path.join(dataset_path, patient)
        if not os.path.isdir(patient_path):
            continue

        files = os.listdir(patient_path)
        for file in files:
            img = nib.load(os.path.join(patient_path, file)).get_fdata()
            if 't2' in file.lower():
                t2_data.append(img)
            elif 'cap' in file.lower():
                mask_data.append(img)

    crop_uniform(t2_data)
    crop_uniform(mask_data)
    return np.array(t2_data), np.array(mask_data)
