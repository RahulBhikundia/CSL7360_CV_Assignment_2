# Cost-Efficient Image Segmentation

A lightweight deep learning-based image segmentation system that balances segmentation quality, single-core CPU inference speed, and model storage footprint under strict computational constraints.

The project implements a compact **SmallUNet** architecture from scratch using PyTorch, trained on the DUTS training dataset (`DUTS-TR`) and evaluated on the held-out validation split and the official `DUTS-TE` test benchmark. The model is additionally tested on six personally captured campus and hostel photographs to examine its generalization on real-world scenes.

A trained model checkpoint (`checkpoints/best_model_4.pth`) is included, allowing users to perform segmentation on their own images without reproducing the training process.

## Directory Structure

```
Assignment 2/
├── captured/
│   ├── original/              # Original personally captured images
│   ├── images_256/            # Resized 256 x 256 input images (PNG)
│   └── masks_256/             # Corresponding 256 x 256 ground-truth masks (PNG)
│
├── checkpoints/
│   └── best_model_4.pth       # Trained model checkpoint (129,326 bytes)
│
├── datasets/
│   ├── DUTS-TR/
│   └── DUTS-TE/
│
├── predictions/               # Generated segmentation masks
├── ground_truth/              # Original manually annotated masks
├── visualizations/            # 4-panel prediction and ground-truth comparisons
│
├── dataset.py                 # Dataset loading and preprocessing
├── model.py                   # SmallUNet architecture (15,949 parameters)
├── train.py                   # Model training (50 epochs with early stopping)
├── predict.py                 # Single-image CLI inference and timing
├── evaluate.py                # Validation set evaluation
├── evaluation_test.py         # Official DUTS-TE test set evaluation
├── benchmark.py               # Single-threaded CPU inference benchmark
├── visualize.py               # Personal photo evaluation and 4-panel overlays
├── prepare_captured.py        # Preparation of captured images and masks
│
├── requirements.txt
├── Report.tex
├── Report.pdf
└── README.md


```

## Model Architecture & Methodology

### 1. SmallUNet

The network follows a 3-stage U-Net encoder-decoder structure built with depthwise-separable convolutions (`3x3` depthwise + `1x1` pointwise), batch normalization, in-place ReLU activations, `2x2` max pooling, and `2x` bilinear upsampling:

* **Encoder (`enc1`–`enc3`):** Progressively extracts visual features across channel widths `3 -> 8 -> 16 -> 32` while downsampling spatial dimensions from `256 x 256` to `32 x 32`.

* **Bottleneck:** Processes the `32 x 32` feature map (`32 -> 48` channels).

* **Decoder (`dec1`–`dec3`) & Skip Connections:** Bilinearly upsamples features and concatenates skip connections from the encoder (`80 -> 32`, `48 -> 16`, and `24 -> 8` channels), followed by a `1 x 1` output convolution (`8 -> 1` channel) at full `256 x 256` resolution.

* **Total Trainable Parameters:** **15,949** (`129,326 bytes` on disk).

### 2. Training Configuration

| Parameter | Configuration | 
| ----- | ----- | 
| Training dataset | DUTS-TR | 
| Training samples | 9,497 | 
| Validation samples | 1,056 | 
| Train-validation split | 90:10 (Seed = 42) | 
| Input / Operating resolution | 256 × 256 | 
| Batch size | 16 | 
| Loss function | BCEWithLogitsLoss | 
| Optimizer | Adam (lr = 0.001) | 
| Max epochs / Early stopping | 50 epochs (patience = 7) | 
| Best validation loss | 0.2603 (Epoch 44) | 

## Requirements & Installation

* **Python:** 3.12 recommended

* **Framework:** PyTorch 2.5.1

* **Image processing:** OpenCV, NumPy, Pillow, Matplotlib

### 1. Create a Virtual Environment

**Windows (Git Bash):**

```
python -m venv .venv
source .venv/Scripts/activate


```

**Windows (PowerShell):**

```
python -m venv .venv
.venv\Scripts\Activate.ps1


```

### 2. Install Dependencies

```
python -m pip install --upgrade pip
pip install -r requirements.txt


```

### 3. Install PyTorch

**CPU-only installation:**

```
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu


```

**NVIDIA GPU installation (CUDA 12.1):**

```
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121


```

## How to Run

### 1. Perform Segmentation on an Individual Image

```
python predict.py --image captured/original/photo1.jpeg --out predictions/predicted_mask1.png


```

Example output:

```
T_CPU_ms: 33.59
S_disk_bytes: 129326


```

### 2. Reproduce Training

```
python train.py


```

The best checkpoint is saved under `checkpoints/best_model_4.pth`.

### 3. Evaluation and Benchmarking

```
python evaluate.py
python evaluation_test.py
python benchmark.py
python prepare_captured.py
python visualize.py


```

## Evaluation & Benchmark Performance

The combined quality score $Q$ is defined as:

$$
Q = 0.70 \times \text{mIoU} + 0.30 \times \text{Boundary F1}
$$

### 1. Segmentation Performance

| Dataset | Images | BG IoU | FG IoU | mIoU | Boundary F1 | Q | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| DUTS-TR Validation | 1,056 | 0.8565 | 0.6519 | 0.7542 | 0.4006 | 0.6481 | 
| DUTS-TE Test | 5,019 | 0.8553 | 0.4273 | 0.6413 | 0.3045 | 0.5403 | 
| Captured Images | 6 | 0.9039 | 0.5344 | 0.7192 | 0.3866 | 0.6194 | 

### 2. Computational Efficiency (DUTS-TE Official Benchmark)

| Metric | Result | 
| ----- | ----- | 
| Model trainable parameters | 15,949 | 
| Checkpoint size ($S_{\text{disk}}$) | 129,326 bytes (0.1233 MiB) | 
| Average CPU inference time ($T_{\text{CPU}}$) | 25.2291 ms | 
| Median CPU inference time | 31.2500 ms | 
| Maximum CPU inference time | 62.5000 ms | 
| Cost-Efficient Segmentation Score (CES) | **0.48408** | 
