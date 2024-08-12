import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os

# Define CapsNet Model
class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, in_channels, out_channels, kernel_size, stride=1, routing_iterations=3):
        super(CapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.routing_iterations = routing_iterations
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
            for _ in range(num_capsules)
        ])
    
    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(u.size(0), self.num_capsules, self.out_channels, -1)
        
        norms = torch.norm(u, dim=-1, keepdim=True)
        squash = (norms**2 / (1 + norms**2)) / (1 + norms**2)
        return squash * u / norms

class PrimaryCapsuleLayer(nn.Module):
    def __init__(self, num_capsules, in_channels, out_channels, kernel_size, stride=1):
        super(PrimaryCapsuleLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.num_capsules = num_capsules
        self.out_channels = out_channels
    
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = x.view(x.size(0), self.num_capsules, self.out_channels, -1)
        norms = torch.norm(x, dim=-1, keepdim=True)
        squash = (norms**2 / (1 + norms**2)) / (1 + norms**2)
        return squash * x / norms

class CapsNet(nn.Module):
    def __init__(self, num_classes):
        super(CapsNet, self).__init__()
        self.primary_capsules = PrimaryCapsuleLayer(num_capsules=8, in_channels=3, out_channels=16, kernel_size=9, stride=2)
        self.secondary_capsules = CapsuleLayer(num_capsules=num_classes, in_channels=16, out_channels=16, kernel_size=9, stride=2)
        self.fc = nn.Linear(16 * num_classes, num_classes)
    
    def forward(self, x):
        x = self.primary_capsules(x)
        x = self.secondary_capsules(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Define your CapsNet model
num_classes = len(categories)  # Define number of classes based on your dataset
model = CapsNet(num_classes=num_classes).to(device)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define dataset and dataloader
dataset = ExcelImageDataset('./dataRef/release_midas.xlsx', root_dirs, transform)
augmented_dataset = AugmentedImageDataset(dataset, './augmented_images', transform)

# Split dataset
train_size = int(0.8 * len(augmented_dataset))
test_size = len(augmented_dataset) - train_size
train_dataset, test_dataset = random_split(augmented_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):  # Adjust epoch count as needed
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0

print('Finished Training')

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on the test dataset: {100 * correct / total:.2f}%')
