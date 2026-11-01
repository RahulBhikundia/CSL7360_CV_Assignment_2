
from model import SmallUNet
import numpy as np
import time
import torch
import cv2
import os

# Set thread limits before importing numerical libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"


# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMAGE_DIR = "datasets/DUTS-TE/DUTS-TE-Image"
CHECKPOINT = "checkpoints/best_model_4.pth"

IMAGE_SIZE = (256, 256)
EXPECTED_IMAGES = 5019

device = torch.device("cpu")

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
cv2.setNumThreads(1)


# --------------------------------------------------
# Load model (not timed)
# --------------------------------------------------

model = SmallUNet().to(device)

model.load_state_dict(
    torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("Model loaded successfully.")


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

def prepare_input(image_bgr):

    # Colour conversion is timed
    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    image_tensor = torch.from_numpy(
        image_rgb.copy()
    ).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    return image_tensor


# --------------------------------------------------
# Collect image paths
# --------------------------------------------------

image_paths = sorted([
    os.path.join(IMAGE_DIR, filename)
    for filename in os.listdir(IMAGE_DIR)
    if filename.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp")
    )
])

if len(image_paths) != EXPECTED_IMAGES:
    raise RuntimeError(
        f"Expected {EXPECTED_IMAGES} images, "
        f"but found {len(image_paths)}."
    )

print("Total test images:", len(image_paths))


# --------------------------------------------------
# Warm-up (not timed)
# --------------------------------------------------

with torch.inference_mode():

    for path in image_paths[:10]:

        image = cv2.imread(path)

        if image is None:
            raise RuntimeError(f"Cannot read image: {path}")

        image = cv2.resize(
            image,
            IMAGE_SIZE,
            interpolation=cv2.INTER_AREA
        )

        tensor = prepare_input(image)

        logits = model(tensor)

        prediction = (
            (logits >= 0)
            .squeeze()
            .numpy()
            .astype(np.uint8)
        )


# --------------------------------------------------
# Official timing
# --------------------------------------------------

times = []

with torch.inference_mode():

    for index, path in enumerate(image_paths):

        # File reading is excluded
        image = cv2.imread(path)

        if image is None:
            raise RuntimeError(f"Cannot read image: {path}")

        # Resizing the original image is excluded
        image = cv2.resize(
            image,
            IMAGE_SIZE,
            interpolation=cv2.INTER_AREA
        )

        # Start timing
        start = time.process_time()

        tensor = prepare_input(image)

        logits = model(tensor)

        prediction = (
            (logits >= 0)
            .squeeze()
            .numpy()
            .astype(np.uint8)
        )

        # Stop timing
        end = time.process_time()

        times.append((end - start) * 1000)

        if (index + 1) % 500 == 0:
            print(f"Processed {index + 1}/{EXPECTED_IMAGES}")


# --------------------------------------------------
# Results
# --------------------------------------------------

average_time = np.mean(times)
median_time = np.median(times)
maximum_time = np.max(times)

model_size = os.path.getsize(CHECKPOINT)

print("\n========== OFFICIAL CPU BENCHMARK ==========")
print(f"Images processed: {len(times)}")
print(f"Average inference time: {average_time:.4f} ms")
print(f"Median inference time: {median_time:.4f} ms")
print(f"Maximum inference time: {maximum_time:.4f} ms")
print(f"Model size: {model_size} bytes")
print("============================================")
