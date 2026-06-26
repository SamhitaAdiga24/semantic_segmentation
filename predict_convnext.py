import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from convnext_model import get_model

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load Model
# -----------------------------
model = get_model().to(device)

model.load_state_dict(
    torch.load(
        "saved_models/convnext.pth",
        map_location=device
    )
)

model.eval()

# -----------------------------
# Load Image
# -----------------------------
image_path = "input/original_images/" + sorted(os.listdir("input/original_images"))[0]
mask_path = "input/masked_images/" + sorted(os.listdir("input/masked_images"))[0]

image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_resized = cv2.resize(image_rgb, (256, 256))

x = torch.tensor(
    image_resized,
    dtype=torch.float32
).permute(2, 0, 1).unsqueeze(0) / 255.0

x = x.to(device)

# -----------------------------
# Prediction
# -----------------------------
with torch.no_grad():
    output = model(x)

pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

# -----------------------------
# Ground Truth
# -----------------------------
gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

gt_mask = cv2.resize(
    gt_mask,
    (256, 256),
    interpolation=cv2.INTER_NEAREST
)

# -----------------------------
# Display Results
# -----------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image_resized)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(gt_mask)
plt.title("Ground Truth")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(pred_mask)
plt.title("ConvNeXt Prediction")
plt.axis("off")

plt.tight_layout()
plt.show()