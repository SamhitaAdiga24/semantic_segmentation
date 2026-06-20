import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

from pspnet_model import get_model

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load Model
# -----------------------------
model = get_model().to(device)

model.load_state_dict(
    torch.load("saved_models/pspnet_50ep.pth")
)

model.eval()

# -----------------------------
# Class Mapping
# -----------------------------
mapping = {
    89: 0,
    104: 1,
    106: 2,
    169: 3,
    184: 4
}

NUM_CLASSES = 5

# -----------------------------
# Metrics Variables
# -----------------------------
total_correct = 0
total_pixels = 0

iou_scores = []
dice_scores = []

image_dir = "input/original_images"
mask_dir = "input/masked_images"

files = sorted(os.listdir(image_dir))

# -----------------------------
# Evaluation Loop
# -----------------------------
for file in tqdm(files):

    image_path = os.path.join(image_dir, file)
    mask_path = os.path.join(mask_dir, file)

    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))

    x = torch.tensor(
        image,
        dtype=torch.float32
    ).permute(2, 0, 1).unsqueeze(0) / 255.0

    x = x.to(device)

    # Prediction
    with torch.no_grad():
        output = model(x)

    pred = torch.argmax(
        output,
        dim=1
    ).squeeze().cpu().numpy()

    # Ground Truth
    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    mask = cv2.resize(
        mask,
        (256, 256),
        interpolation=cv2.INTER_NEAREST
    )

    gt = np.zeros_like(mask)

    for old_val, new_val in mapping.items():
        gt[mask == old_val] = new_val

    # Accuracy
    total_correct += np.sum(pred == gt)
    total_pixels += gt.size

    # IoU and Dice per class
    for cls in range(NUM_CLASSES):

        pred_cls = (pred == cls)
        gt_cls = (gt == cls)

        intersection = np.logical_and(
            pred_cls,
            gt_cls
        ).sum()

        union = np.logical_or(
            pred_cls,
            gt_cls
        ).sum()

        if union > 0:
            iou_scores.append(
                intersection / union
            )

        denom = pred_cls.sum() + gt_cls.sum()

        if denom > 0:
            dice_scores.append(
                (2 * intersection) / denom
            )

# -----------------------------
# Final Metrics
# -----------------------------
accuracy = total_correct / total_pixels
miou = np.mean(iou_scores)
dice = np.mean(dice_scores)

print("\n===== RESULTS =====")
print(f"Pixel Accuracy : {accuracy*100:.2f}%")
print(f"Mean IoU       : {miou:.4f}")
print(f"Dice Score     : {dice:.4f}")