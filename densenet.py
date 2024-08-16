import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import optuna
from collections import Counter
import cv2

# Define categories and image size
categories = ['7-malignant-bcc', '1-benign-melanocytic nevus', '6-benign-other',
              '14-other-non-neoplastic/inflammatory/infectious', '8-malignant-scc',
              '9-malignant-sccis', '10-malignant-ak', '3-benign-fibrous papule',
              '4-benign-dermatofibroma', '2-benign-seborrheic keratosis',
              '5-benign-hemangioma', '11-malignant-melanoma',
              '13-other-melanocytic lesion with possible re-excision (severe/spitz nevus, aimp)',
              '12-malignant-other']
img_size = 224

# Define transformations
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class ExcelImageDataset(Dataset):
    def __init__(self, excel_file, root_dirs, transform=None):
        self.data_frame = pd.read_excel(excel_file)
        self.data_frame.iloc[:, 0] = self.data_frame.iloc[:, 0].astype(str)
        self.root_dirs = root_dirs
        self.transform = transform
        self.label_map = {label: idx for idx, label in enumerate(categories)}
        self.image_paths = self._get_image_paths()

    def _get_image_paths(self):
        valid_paths = []
        for idx, row in self.data_frame.iterrows():
            img_found = False
            for root_dir in self.root_dirs:
                img_name = os.path.join(root_dir, row['midas_file_name'])
                if os.path.isfile(img_name):
                    label = row['clinical_impression_1']
                    if label not in self.label_map:
                        print(f"Warning: Label '{label}' not in label_map.")
                        continue
                    valid_paths.append((img_name, label))
                    img_found = True
                    break
            if not img_found:
                print(f"Warning: Image {row['midas_file_name']} not found in any root directory.")
        print(f"Total valid paths found: {len(valid_paths)}")
        return valid_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name, label = self.image_paths[idx]
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.label_map.get(label, -1), dtype=torch.long)
        return image, label

# Define the root directories
root_dirs = [
    '/root/stanfordData4321/stanfordData4321/images2',
    '/root/stanfordData4321/stanfordData4321/images1',
    '/root/stanfordData4321/stanfordData4321/images3',
    '/root/stanfordData4321/stanfordData4321/images4'
]

# Function to count images per label
def count_images_per_label(dataset):
    label_counts = Counter(label.item() for _, label in dataset)
    return {categories[label]: count for label, count in label_counts.items()}

# Load dataset
dataset = ExcelImageDataset('./dataRef/release_midas.xlsx', root_dirs, transform)

# Function to count images per label in augmented dataset
class AugmentedImageDataset(Dataset):
    def __init__(self, original_dataset, augmented_dir, transform=None):
        self.original_dataset = original_dataset
        self.augmented_dir = augmented_dir
        self.transform = transform
        self.augmented_paths = self._get_augmented_paths()

    def _get_augmented_paths(self):
        augmented_paths = []
        for root, _, files in os.walk(self.augmented_dir):
            for file in files:
                if file.endswith(".png"):
                    img_path = os.path.join(root, file)
                    label = int(os.path.basename(root))
                    augmented_paths.append((img_path, label))
        return augmented_paths

    def __len__(self):
        return len(self.augmented_paths)

    def __getitem__(self, idx):
        img_path, label = self.augmented_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

# Create the combined dataset using augmented images
augmented_dataset = AugmentedImageDataset(dataset, './augmented_images', transform)
print(f"Total images in augmented dataset: {len(augmented_dataset)}")

# Split dataset
train_size = int(0.8 * len(augmented_dataset))
test_size = len(augmented_dataset) - train_size
train_dataset, test_dataset = random_split(augmented_dataset, [train_size, test_size])

print(f"Train dataset length: {len(train_dataset)}, Test dataset length: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Load pre-trained DenseNet model and modify the final layer
from torchvision.models import densenet201, DenseNet201_Weights

weights = DenseNet201_Weights.DEFAULT
net = densenet201(weights=weights)
num_ftrs = net.classifier.in_features
net.classifier = nn.Linear(num_ftrs, len(categories))

# Move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
print(device)

# Optuna optimization
def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    momentum = trial.suggest_float('momentum', 0.5, 0.9)
    
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1):  # Fewer epochs for faster optimization
        net.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1)

best_params = study.best_params
print("Best parameters found by Optuna:", best_params)

# Training with the best parameters
best_lr = best_params['lr']
best_momentum = best_params['momentum']
optimizer = optim.SGD(net.parameters(), lr=best_lr, momentum=best_momentum)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()

for epoch in range(10):  # Adjust epoch count as needed
    net.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')

# Evaluate the model
net.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)  # Move labels to the same device as images
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on the test dataset: {100 * correct / total:.2f}%')

def occlusion_sensitivity(model, image_tensor, label, patch_size=15, stride=8, output_path=None):
    model.eval()
    c, h, w = image_tensor.size()
    sensitivity_map = torch.zeros(h, w)

    # Get the model's original prediction
    original_output = model(image_tensor.unsqueeze(0))
    original_prediction = torch.argmax(original_output, dim=1).item()

    for i in range(0, w, stride):
        for j in range(0, h, stride):
            occluded_image = image_tensor.clone()
            occluded_image[:, j:j+patch_size, i:i+patch_size] = 0  # Occlude a patch

            # Get the model's prediction for the occluded image
            output = model(occluded_image.unsqueeze(0))
            prediction = torch.argmax(output, dim=1).item()

            # Update the sensitivity map
            sensitivity_map[j:j+patch_size, i:i+patch_size] = float(prediction != original_prediction)

    sensitivity_map = sensitivity_map / sensitivity_map.max()  # Normalize the sensitivity map

    if output_path:
        plt.imshow(sensitivity_map, cmap='hot', interpolation='nearest')
        plt.savefig(output_path)
        print(f"Sensitivity map saved to {output_path}")
    return sensitivity_map

# Load and preprocess the image
image_path = '/root/stanfordData4321/stanfordData4321/images4/s-prd-784541963.jpg'
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).to(device)  # Apply transformations and move to device

# Generate the occlusion sensitivity map
occlusion_sensitivity(
    net,
    image_tensor,
    label=None,  # Replace with the actual label if needed
    patch_size=30,  # Size of the occlusion patch
    stride=15,      # Stride for moving the patch
    output_path='./occlusion_sensitivity_map.png'  # Save image to a file
)

