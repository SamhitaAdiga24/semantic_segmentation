import cv2
import torch
import numpy as np
from convnext_model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model().to(device)
model.load_state_dict(torch.load("saved_models/convnext.pth"))
model.eval()

image_path = "input/original_images/" + sorted(__import__("os").listdir("input/original_images"))[0]

image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

img = cv2.resize(image_rgb, (256, 256))

x = torch.tensor(
    img,
    dtype=torch.float32
).permute(2,0,1).unsqueeze(0) / 255.0

x = x.to(device)

with torch.no_grad():
    output = model(x)

pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

colors = np.array([
    [0, 0, 0],        # Class 0
    [0, 255, 0],      # Class 1
    [255, 0, 0],      # Class 2
    [255, 255, 0],    # Class 3
    [255, 0, 255]     # Class 4
], dtype=np.uint8)

colored_mask = colors[pred]

cv2.imwrite(
    "prediction_color.png",
    cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
)
print("Saved as prediction.png")