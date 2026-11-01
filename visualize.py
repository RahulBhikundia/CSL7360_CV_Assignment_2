
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from evaluate import boundary_f1


# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMAGE_DIR = "captured/original"
GT_DIR = "captured/masks_256"
PRED_DIR = "predictions"
OUTPUT_DIR = "visualizations"

IMAGE_SIZE = (256, 256)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# Metric accumulation
# --------------------------------------------------

TP = 0
TN = 0
FP = 0
FN = 0

bf1_scores = []
processed = 0


# --------------------------------------------------
# Process all six personal photos
# --------------------------------------------------

for i in range(1, 7):

    image_path = os.path.join(
        IMAGE_DIR, f"photo{i}.jpeg"
    )

    gt_path = os.path.join(
        GT_DIR, f"photo{i}.png"
    )

    pred_path = os.path.join(
        PRED_DIR, f"predicted_mask{i}.png"
    )

    # Load files
    image = cv2.imread(image_path)

    prediction = cv2.imread(
        pred_path,
        cv2.IMREAD_GRAYSCALE
    )

    ground_truth = cv2.imread(
        gt_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise FileNotFoundError(image_path)

    if prediction is None:
        raise FileNotFoundError(pred_path)

    if ground_truth is None:
        raise FileNotFoundError(gt_path)

    # Resize image and masks
    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    prediction = cv2.resize(
        prediction,
        IMAGE_SIZE,
        interpolation=cv2.INTER_NEAREST
    )

    ground_truth = cv2.resize(
        ground_truth,
        IMAGE_SIZE,
        interpolation=cv2.INTER_NEAREST
    )

    # Convert masks to binary
    pred = prediction >= 128
    gt = ground_truth >= 128

    # Update global confusion matrix
    TP += np.sum(pred & gt)
    TN += np.sum(~pred & ~gt)
    FP += np.sum(pred & ~gt)
    FN += np.sum(~pred & gt)

    # Calculate Boundary F1 for this image
    score = boundary_f1(pred, gt)
    bf1_scores.append(score)

    processed += 1

    # --------------------------------------------------
    # Generate semi-transparent overlays
    # --------------------------------------------------

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Ground-truth overlay: red
    gt_overlay = image_rgb.copy()

    gt_overlay[gt] = (
        0.5 * image_rgb[gt].astype(np.float32)
        + 0.5 * np.array([255, 0, 0])
    ).astype(np.uint8)

    # Prediction overlay: green
    pred_overlay = image_rgb.copy()

    pred_overlay[pred] = (
        0.5 * image_rgb[pred].astype(np.float32)
        + 0.5 * np.array([0, 255, 0])
    ).astype(np.uint8)

    # --------------------------------------------------
    # Plot comparison
    # --------------------------------------------------

    fig, axes = plt.subplots(
        1, 4, figsize=(16, 4)
    )

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")

    axes[1].imshow(gt_overlay)
    axes[1].set_title("Ground Truth Overlay (Red)")

    axes[2].imshow(pred_overlay)
    axes[2].set_title("Prediction Overlay (Green)")

    # Side-by-side comparison using both masks:
    # Red = ground truth only
    # Green = prediction only
    # Yellow = overlap
    comparison = np.zeros(
        (256, 256, 3),
        dtype=np.uint8
    )

    comparison[gt & ~pred] = [255, 0, 0]
    comparison[pred & ~gt] = [0, 255, 0]
    comparison[gt & pred] = [255, 255, 0]

    axes[3].imshow(comparison)
    axes[3].set_title("Mask Comparison")

    for ax in axes:
        ax.axis("off")

    plt.suptitle(
        f"Photo {i} | Boundary F1: {score:.4f}"
    )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"photo{i}_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Photo {i}: Boundary F1 = {score:.4f}"
    )


# --------------------------------------------------
# Calculate aggregate metrics
# --------------------------------------------------

if processed == 0:
    raise RuntimeError("No photos were processed.")

background_denominator = TN + FP + FN
foreground_denominator = TP + FP + FN

iou_background = (
    TN / background_denominator
    if background_denominator > 0 else 0.0
)

iou_foreground = (
    TP / foreground_denominator
    if foreground_denominator > 0 else 0.0
)

miou = (iou_background + iou_foreground) / 2

# Average per-image Boundary F1
bf1 = np.mean(bf1_scores)

Q = 0.70 * miou + 0.30 * bf1


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\n========== PERSONAL PHOTO EVALUATION ==========")

print("Images processed:", processed)
print("Background IoU:", iou_background)
print("Foreground IoU:", iou_foreground)
print("mIoU:", miou)
print("Boundary F1:", bf1)
print("Q:", Q)

print("\nComparison images saved in:", OUTPUT_DIR)
