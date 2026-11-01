
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from model import SmallUNet
from dataset import DUTSDataset


os.makedirs("checkpoints", exist_ok=True)

best_val_loss = float("inf")

# Load dataset
dataset = DUTSDataset(
    "datasets/DUTS-TR/DUTS-TR-Image",
    "datasets/DUTS-TR/DUTS-TR-Mask"
)

# Reproduce the same 90/10 split
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

# Select device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

model = SmallUNet().to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

epochs = 50  # Increased to allow the model to fully converge
patience = 7
epochs_no_improve = 0
best_val_loss = float('inf')

checkpoint_path = "checkpoints/best_model_4.pth"

for epoch in range(epochs):

    # Training
    model.train()
    running_loss = 0.0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    train_loss = running_loss / len(train_dataset)

    # Validation
    model.eval()
    val_running_loss = 0.0

    with torch.inference_mode():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            val_running_loss += loss.item() * images.size(0)

    val_loss = val_running_loss / len(val_dataset)

    # Save best checkpoint & Early Stopping Logic
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        epochs_no_improve = 0

        torch.save(
            model.state_dict(),
            checkpoint_path
        )

        print("Best model saved!")
    else:
        epochs_no_improve += 1

    print(
        f"Epoch [{epoch + 1}/{epochs}]",
        f"Training Loss: {train_loss:.4f}",
        f"Validation Loss: {val_loss:.4f}"
    )

    if epochs_no_improve >= patience:
        print(f"Early stopping triggered! No improvement for {patience} consecutive epochs.")
        break

print("Training completed.")
print("Best validation loss:", best_val_loss)
print("Checkpoint:", checkpoint_path)