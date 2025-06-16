import matplotlib.pyplot as plt
import numpy as np

def show_overlay(image, mask=None, threshold=0.5, alpha=0.4, ax=None, title=None):
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image.squeeze()
    if mask is not None and mask.ndim == 3:
        mask = mask.squeeze()

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        own_fig = True
    else:
        own_fig = False

    ax.imshow(image, cmap='gray')
    ax.axis('off')

    if mask is not None:
        binary_mask = (mask > threshold).astype(np.uint8)
        overlay = np.zeros((*binary_mask.shape, 4))
        overlay[binary_mask == 1] = [1, 0, 0, alpha]  # red with alpha
        ax.imshow(overlay)

    if title:
        ax.set_title(title)
    if own_fig:
        plt.show()

def compare_predictions(images, ground_truths, predictions, indices=None, threshold=0.5):
    if indices is None:
        indices = list(range(min(6, len(images))))

    fig, axs = plt.subplots(3, len(indices), figsize=(4 * len(indices), 9))

    for idx, i in enumerate(indices):
        show_overlay(images[i], ax=axs[0, idx], title=f"Input {i}")
        show_overlay(images[i], ground_truths[i], ax=axs[1, idx], title="Ground Truth")
        show_overlay(images[i], predictions[i], threshold=threshold, ax=axs[2, idx], title="Prediction")

    plt.tight_layout()
    plt.show()