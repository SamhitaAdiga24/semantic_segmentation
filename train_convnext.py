import torch
from torch.utils.data import DataLoader
from dataset import LandDataset
from convnext_model import get_model

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# Dataset
dataset = LandDataset(
    "input/original_images",
    "input/masked_images"
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

# Model
model = get_model().to(device)

# Loss
loss_fn = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

# Epochs
epochs = 50

# Training Loop
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

    # Calculate average loss for this epoch
    avg_loss = total_loss / len(loader)

    print(
        f"Epoch {epoch+1}/{epochs}, Loss={avg_loss:.4f}"
    )

# Save Model
torch.save(
    model.state_dict(),
    "saved_models/convnext.pth"
)

print("Model Saved")