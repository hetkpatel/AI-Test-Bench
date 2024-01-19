import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os

# create director called logs if it does not exist
if not os.path.exists("./logs"):
    os.mkdir("logs")

# check if file exists and delete it
if os.path.exists("./logs/cifar10_alpha.log"):
    os.remove("./logs/cifar10_alpha.log")

writer = SummaryWriter("runs/cifar10_alpha")

# Load the CIFAR10 data
t = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
train_dataset = datasets.CIFAR10(root="./data", download=True, train=True, transform=t)
train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)


# Define the Convolutional Neural Network architecture
class CIFAR_Convo(nn.Module):
    def __init__(self):
        super(CIFAR_Convo, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=6,
            kernel_size=5,
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=(1, 1),
            padding=(1, 1),
        )
        self.fc_lyr1 = nn.Linear(16 * 6 * 6, 128)
        self.fc_lyr2 = nn.Linear(128, 64)
        self.fc_lyr3 = nn.Linear(64, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.pool(out)
        out = F.relu(self.conv2(out))
        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = F.relu(self.fc_lyr1(out))
        out = F.relu(self.fc_lyr2(out))
        out = self.fc_lyr3(out)

        return out


# Create an instance of the CNN model
model = CIFAR_Convo()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

# Train the model
model.train()
epoch = 1
for epoch in range(epoch):
    for i, (X_train_tensor, y_train_tensor) in enumerate(train_loader):
        # Forward pass
        y_pred = model(X_train_tensor)
        # Compute Loss
        loss = criterion(y_pred.squeeze(), y_train_tensor)

        # Zero grad
        optimizer.zero_grad()
        # Backward pass
        loss.backward()
        optimizer.step()

        # Output to tensorboard
        writer.add_scalar(
            tag="Loss",
            scalar_value=loss.item(),
            global_step=epoch * len(train_loader) + i,
            new_style=True,
            double_precision=True,
        )

# Save the model
torch.save(model, "cifar10_model.pt")

writer.close()
