import pandas as pd
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import optuna
from sklearn.model_selection import train_test_split

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_excel(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if idx >= len(self.data_frame):
            raise IndexError(f"Index {idx} is out of range.")
        img_name = os.path.join(self.data_frame.iloc[idx, 2])
        image = Image.open(img_name)
        label = self.data_frame.iloc[idx, 18]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load data
csv_file = '/root/stanfordData4321/stanfordData4321-1/dataRef/release_midas.xlsx'
dataset = CustomDataset(csv_file=csv_file, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Define the model
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 15)  # Assuming 15 classes

    def forward(self, x):
        return self.model(x)

# Training function
def train_model(lr, momentum):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    for epoch in range(1):
        net.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f'Epoch [{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    # Evaluate model
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

# Objective function for Optuna
def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    momentum = trial.suggest_uniform('momentum', 0.0, 1.0)
    accuracy = train_model(lr, momentum)
    return accuracy

# Optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1)  # Adjust the number of trials as needed

# Print best trial
print(f'Best trial:')
print(f'  Value: {study.best_value}')
print(f'  Params: ')
for key, value in study.best_params.items():
    print(f'    {key}: {value}')
