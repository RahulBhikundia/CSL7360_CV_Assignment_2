import os
import cv2
import numpy as np

os.makedirs("captured/images_256", exist_ok=True)
os.makedirs("captured/masks_256", exist_ok=True)

for i in range(1, 7):

    # Original photograph
    image_path = f"captured/original/photo{i}.jpeg"

    # Original manually annotated mask
    mask_path = f"ground_truth/photo_mask{i}.png"

    # Handle JPEG masks for photographs 2-6
    if not os.path.exists(mask_path):
        mask_path = f"ground_truth/photo_mask{i}.jpeg"

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(image_path)

    if mask is None:
        raise FileNotFoundError(mask_path)

    # Resize image
    image_256 = cv2.resize(
        image,
        (256, 256),
        interpolation=cv2.INTER_AREA
    )

    # Resize mask
    mask_256 = cv2.resize(
        mask,
        (256, 256),
        interpolation=cv2.INTER_NEAREST
    )

    # Convert mask to binary 0/255
    mask_256 = np.where(mask_256 >= 128, 255, 0).astype(np.uint8)

    cv2.imwrite(
        f"captured/images_256/photo{i}.png",
        image_256
    )

    cv2.imwrite(
        f"captured/masks_256/photo{i}.png",
        mask_256
    )

    print(f"Prepared photograph {i}")

print("All six photographs prepared successfully.")