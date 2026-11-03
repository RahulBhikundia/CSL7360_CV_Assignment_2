import torch
import numpy as np

from torch.utils.data import DataLoader, random_split
from model import SmallUNet
from dataset import DUTSDataset

def get_boundary(mask):
    mask = mask.astype(bool)

    neighbor_background = np.zeros_like(mask)

    neighbor_background[1:, :] |= ~mask[:-1, :]
    neighbor_background[:-1, :] |= ~mask[1:, :]
    neighbor_background[:, 1:] |= ~mask[:, :-1]
    neighbor_background[:, :-1] |= ~mask[:, 1:]
    
    neighbor_background[0, :] = True
    neighbor_background[-1, :] = True
    neighbor_background[:, 0] = True
    neighbor_background[:, -1] = True


    return mask & neighbor_background 

def dilate_5x5(boundary):

    padded = np.pad(boundary, 2, mode='constant')
    result = np.zeros_like(boundary, dtype=bool)

    for dy in range(5):
        for dx in range(5):

            result |= padded[
                dy:dy + boundary.shape[0],
                dx:dx + boundary.shape[1]
            ]

    return result

def boundary_f1(prediction, target):

    pred_boundary = get_boundary(prediction)
    target_boundary = get_boundary(target)

    pred_count = pred_boundary.sum()
    target_count = target_boundary.sum()

    if pred_count == 0 and target_count == 0:
        return 1.0

    if pred_count == 0 or target_count == 0:
        return 0.0

    target_dilated = dilate_5x5(target_boundary)
    pred_dilated = dilate_5x5(pred_boundary)

    matched_pred = np.sum(pred_boundary & target_dilated)
    matched_target = np.sum(target_boundary & pred_dilated)

    precision = matched_pred/pred_count
    recall = matched_target/target_count

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
dataset = DUTSDataset(
    "datasets/DUTS-TE/DUTS-TE-Image",
    "datasets/DUTS-TE/DUTS-TE-Mask"
)

test_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

# Load trained model
model = SmallUNet().to(device)

model.load_state_dict(
    torch.load(
        "checkpoints/best_model_4.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("Model loaded succesfully!")

TN = 0
TP = 0
FN = 0
FP = 0

with torch.inference_mode():
    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)

        predictions = (logits >= 0)
        targets = (masks >= 0.5)

        TN += ((~predictions) & (~targets)).sum().item()
        FP += ((predictions) & (~targets)).sum().item()
        FN += ((~predictions) & (targets)).sum().item()
        TP += ((predictions) & (targets)).sum().item()

iou_background = TN / (TN + FP + FN)
iou_foreground = TP / (TP + FP + FN)

miou = (iou_background + iou_foreground) / 2

print("Background IoU:", iou_background)
print("Foreground IoU:", iou_foreground)
print("mIoU:", miou)

bf1_scores = []

with torch.inference_mode():

    for images, masks in test_loader:
        
        images = images.to(device)

        logits = model(images)
        predictions = (logits >= 0).cpu().numpy()

        targets = (masks >= 0.5).numpy()

        for i in range(images.size(0)):

            pred = predictions[i, 0]
            target = targets[i, 0]

            score = boundary_f1(pred, target)
            bf1_scores.append(score)

bf1 = np.mean(bf1_scores)

print("Boundary F1:", bf1)

Q = 0.70 * miou + 0.30 * bf1

print("Q:", Q)