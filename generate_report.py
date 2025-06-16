from PIL import Image
import os

# Paths
IMG_DIR = 'data/test_patient/slices/'
OUTPUT_PDF = 'data/test_patient/prostate_prediction_report.pdf'

# Load all .png files and sort them
image_files = sorted(
    [f for f in os.listdir(IMG_DIR) if f.endswith('.png')],
    key=lambda x: int(x.split('_')[1].split('.')[0])  # Extract slice number
)

# Load images and convert to RGB
images = [Image.open(os.path.join(IMG_DIR, f)).convert('RGB') for f in image_files]

# Save as multipage PDF
if images:
    images[0].save(OUTPUT_PDF, save_all=True, append_images=images[1:])
    print(f"[INFO] PDF report saved to: {OUTPUT_PDF}")
else:
    print("[WARNING] No PNG images found in the slice_overlays directory.")
