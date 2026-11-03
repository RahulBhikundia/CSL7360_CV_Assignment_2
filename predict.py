import os

# Set CPU threads before importing numerical libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import time
import cv2
import numpy as np
import torch

from model import SmallUNet

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
cv2.setNumThreads(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device("cpu")

    model = SmallUNet().to(device)
    checkpoint = torch.load(
        "checkpoints/best_model_4.pth",
        map_location=device,
        weights_only=True
    )
    model.load_state_dict(checkpoint)
    model.eval()
    with torch.no_grad():
        model = torch.jit.optimize_for_inference(torch.jit.trace(model, torch.randn(1, 3, 256, 256)))

    # Read and resize the image before timing
    image = cv2.imread(args.image)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    image = cv2.resize(
        image,
        (256, 256),
        interpolation=cv2.INTER_AREA
    )

    def inference():
        # Colour conversion and tensor preparation are timed
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.copy())
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0

        with torch.inference_mode():
            logits = model(tensor)
            mask = (logits >= 0).squeeze().numpy().astype(np.uint8)

        return mask

    # Warm-up
    for _ in range(5):
        inference()

    # Measure 20 runs
    times = []

    for _ in range(20):
        start = time.process_time()
        mask = inference()
        end = time.process_time()
        times.append((end - start) * 1000)

    # Save final binary mask as 0 and 255
    output_mask = (mask * 255).astype(np.uint8)

    if not cv2.imwrite(args.out, output_mask):
        raise RuntimeError(f"Could not save output: {args.out}")

    average_time = sum(times) / len(times)

    model_size = os.path.getsize(
        "checkpoints/best_model_4.pth"
    )

    print(f"T_CPU_ms: {average_time:.2f}")
    print(f"S_disk_bytes: {model_size}")


if __name__ == "__main__":
    main()
