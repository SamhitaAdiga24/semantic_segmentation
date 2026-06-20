import torch
from torch.utils.data import DataLoader
from dataset import LandDataset
from deeplab_model import get_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = LandDataset(
    "input/original_images",
    "input/masked_images"
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

model = get_model().to(device)

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

epochs = 50

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for images, masks in loader:

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        loss = loss_fn(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

avg_loss = total_loss / len(loader)

print(
    f"Epoch {epoch+1}/{epochs}, Loss={avg_loss:.4f}"
)

torch.save(
    model.state_dict(),
    "saved_models/deeplabv3plus.pth"
)

print("Model Saved")