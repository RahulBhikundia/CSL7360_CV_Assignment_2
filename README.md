# Cost-Efficient Image Segmentation

A lightweight deep learning-based image segmentation system that aims to balance segmentation quality, inference speed, and model size under computational constraints.

The project implements a compact **SmallUNet** architecture using PyTorch, trained on the DUTS training dataset and evaluated on the DUTS validation and test datasets. The model is additionally tested on personally captured images to examine its performance on real-world scenes.

A trained model checkpoint is included, allowing users to perform segmentation on their own images without having to reproduce the training process.

---

## 📁 Directory Structure

```text
Assignment 2/
├── captured/
│   ├── original/              # Original personally captured images
│   ├── images_256/            # Resized input images
│   └── masks_256/             # Corresponding ground-truth masks
│
├── checkpoints/
│   └── best_model_192.pth     # Trained model checkpoint
│
├── datasets/
│   ├── DUTS-TR/
│   └── DUTS-TE/
│
├── predictions/               # Generated segmentation masks
├── ground_truth/              # Ground-truth masks for evaluation
├── visualizations/            # Prediction and ground-truth comparisons
│
├── dataset.py                 # Dataset loading and preprocessing
├── model.py                   # SmallUNet architecture
├── train.py                   # Model training
├── predict.py                 # Inference on individual images
├── evaluate.py                # Validation evaluation
├── evaluation_test.py         # DUTS-TE evaluation
├── benchmark.py               # CPU inference benchmarking
├── visualize.py               # Segmentation visualizations
├── prepare_captured.py        # Preparation of captured images
│
├── requirements.txt
├── Report.tex
├── Report.pdf
└── README.md
```

---

## 🧠 Model Architecture & Methodology

The project uses a lightweight U-Net-based segmentation architecture designed to reduce computational requirements while retaining useful spatial information.

### 1. SmallUNet

The network follows an encoder-decoder structure:

- **Encoder:** Extracts increasingly abstract visual features while reducing spatial dimensions.
- **Bottleneck:** Processes the high-level feature representation.
- **Decoder:** Recovers spatial resolution to generate a pixel-level segmentation map.
- **Skip Connections:** Transfer spatial information from encoder stages to corresponding decoder stages.

Depthwise-separable convolutions are used to reduce the number of learnable parameters and computational cost.

The model internally processes images at a resolution of 192 × 192 and produces segmentation predictions at 256 × 256.

### 2. Training Configuration

| Parameter | Configuration |
|---|---|
| Training dataset | DUTS-TR |
| Training samples | 9,497 |
| Validation samples | 1,056 |
| Train-validation split | 90:10 |
| Random seed | 42 |
| Batch size | 16 |
| Loss function | BCEWithLogitsLoss |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Training epochs | 10 |

The best-performing checkpoint on the validation set is saved for subsequent evaluation and inference.

---

## ⚙️ Requirements & Installation

The project uses Python and PyTorch for model implementation, training, and inference.

- **Python:** 3.12 recommended
- **Framework:** PyTorch 2.5.1
- **Image processing:** OpenCV, NumPy, Pillow

### 1. Create a Virtual Environment

Navigate to the project directory and create an isolated Python environment.

**Windows (Git Bash):**

```bash
python -m venv .venv
source .venv/Scripts/activate
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Install PyTorch

Install the appropriate PyTorch build for your hardware.

**CPU-only installation:**

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

**NVIDIA GPU installation (CUDA 12.1):**

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

For other configurations, refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

Verify the installation:

```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

---

## 🚀 How to Run

### 1. Perform Segmentation on Your Own Image

The repository includes the trained model checkpoint:

```text
checkpoints/best_model_192.pth
```

Users can directly perform inference without downloading the DUTS datasets or training the model.

1. Place the input image somewhere accessible from the project directory.
2. Execute the prediction script:

```bash
python predict.py --image path/to/input.jpg --out predictions/output_mask.png
```

3. The generated segmentation mask will be saved at:

```text
predictions/output_mask.png
```

The output is a binary PNG image with:

- `255`: Foreground
- `0`: Background

The script performs CPU inference and reports the average inference time and checkpoint size.

Example output:

```text
T CPU ms: 30.47
S disk bytes: 185126
```

Actual inference time depends on the hardware.

### 2. Reproduce Training

To train the model from scratch, arrange the DUTS training dataset as follows:

```text
datasets/
└── DUTS-TR/
    ├── DUTS-TR-Image/
    └── DUTS-TR-Mask/
```

Run:

```bash
python train.py
```

The best checkpoint is saved under:

```text
checkpoints/best_model_192.pth
```

### 3. Evaluation and Benchmarking

The following scripts are provided:

| Script | Description |
|---|---|
| `evaluate.py` | Evaluates the model on the validation split |
| `evaluation_test.py` | Evaluates the model on DUTS-TE |
| `benchmark.py` | Measures CPU inference performance |
| `visualize.py` | Generates segmentation visualizations |
| `prepare_captured.py` | Prepares personally captured images |

Execute the evaluation scripts from the project root:

```bash
python evaluate.py
python evaluation_test.py
python benchmark.py
```

For test-set evaluation, arrange DUTS-TE as follows:

```text
datasets/
└── DUTS-TE/
    ├── DUTS-TE-Image/
    └── DUTS-TE-Mask/
```

---

## 📊 Evaluation & Benchmark Performance

The model is evaluated using the following metrics:

- **mIoU:** Mean Intersection over Union, measuring pixel-level segmentation overlap.
- **Boundary F1:** Measures the quality of predicted object boundaries.
- **Q:** Combined segmentation quality score.

The combined quality score is calculated as:

\[
Q = 0.70 \times \text{mIoU} + 0.30 \times \text{Boundary F1}
\]

### 1. Segmentation Performance

| Dataset | mIoU | Boundary F1 | Q |
|---|---:|---:|---:|
| DUTS Validation | 0.7558 | 0.3871 | 0.6452 |
| DUTS Test | 0.6302 | 0.2882 | 0.5276 |
| Captured Images | 0.7380 | 0.3987 | 0.6362 |

### 2. Computational Performance

| Metric | Result |
|---|---:|
| Model parameters | 29,617 |
| Checkpoint size | 185,126 bytes |
| Average CPU inference time | 45.88 ms |
| Computational Efficiency Score (CES) | 0.42843 |

The CPU benchmark measurements were obtained on the DUTS test dataset. Actual inference time may vary depending on the hardware and execution environment.

Detailed experimental results, visual comparisons, and failure-case analysis are presented in `Report.pdf`.

---

## 📝 Reproducibility Notes

- Run scripts from the project root directory.
- The supplied checkpoint enables inference without downloading the DUTS datasets.
- DUTS-TR is required for training and DUTS-TE is required for test evaluation.
- An NVIDIA GPU is not required for inference.
- The model architecture is defined in `model.py`, while the trained parameters are provided in `checkpoints/best_model_192.pth`.
- The detailed methodology, experimental analysis, and visual results are documented in `Report.pdf`.