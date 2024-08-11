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
from collections import Counter
import optuna

# Print the current working directory
print("Current working directory:", os.getcwd())

# Load the Excel file
excel_file_path = './dataRef/release_midas.xlsx'
if not os.path.exists(excel_file_path):
    raise FileNotFoundError(f"{excel_file_path} does not exist. Please check the path.")

df = pd.read_excel(excel_file_path)
print("Excel file loaded. First few rows:")
print(df.head())

# Define categories and image size
categories = ['7-malignant-bcc', '1-benign-melanocytic nevus', '6-benign-other',
              '14-other-non-neoplastic/inflammatory/infectious', '8-malignant-scc',
              '9-malignant-sccis', '10-malignant-ak', '3-benign-fibrous papule',
              '4-benign-dermatofibroma', '2-benign-seborrheic keratosis',
              '5-benign-hemangioma', '11-malignant-melanoma',
              '13-other-melanocytic lesion with possible re-excision (severe/spitz nevus, aimp)',
              '12-malignant-other']
img_size = 224

# Compose the transformation pipeline
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

        # Ensure the label categories are consistent with the provided categories
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

        # Convert label to tensor with encoding
        label = torch.tensor(self.label_map.get(label, -1), dtype=torch.long)
        return image, label

# Define the root directories
root_dirs = [
    '/root/stanfordData4321/stanfordData4321/images2',
    '/root/stanfordData4321/stanfordData4321/images1',
    '/root/stanfordData4321/stanfordData4321/images3',
    '/root/stanfordData4321/stanfordData4321/images4'
    '/root/stanfordData4321/stanfordData4321/augmented_images'
]

# Function to count images per label
def count_images_per_label(dataset):
    label_counts = Counter(label.item() for _, label in dataset)
    return {categories[label]: count for label, count in label_counts.items()}

# Load the dataset using augmented images from GitHub
dataset = ExcelImageDataset(excel_file_path, root_dirs, transform)
image_counts = count_images_per_label(dataset)
print("Image counts:")
for label, count in image_counts.items():
    print(f"{label}: {count}")

# Split dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Print the sizes of training and test datasets
print(f"Train dataset length: {len(train_dataset)}, Test dataset length: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Define a function to train and evaluate the model
def train_model(trial):
    # Load pre-trained model and modify the final layer
    weights = models.ResNet18_Weights.DEFAULT
    net = models.resnet18(weights=weights)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, len(categories))

    # Move the model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # Define loss function and optimizer
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    momentum = trial.suggest_float("momentum", 0.5, 0.9)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(5):
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
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    # Evaluate the model
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        for data in test_loader:
            images, labels = data
            outputs = net(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Optimize the hyperparameters using Optuna
study = optuna.create_study(direction="maximize")
study.optimize(train_model, n_trials=2)

# Print the best hyperparameters
print("Best hyperparameters: ", study.best_params)

# Load the best model and apply Grad-CAM
def apply_grad_cam(img_path, model, transform, target_layer):
    model.eval()
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(dim=1)

    grad_cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=True)
    grayscale_cam = grad_cam(input_tensor=input_tensor, target_category=pred.item())[0, :]
    cam_image = show_cam_on_image(np.array(image) / 255.0, grayscale_cam, use_rgb=True)

    plt.imshow(cam_image)
    plt.show()

# Example usage of Grad-CAM
best_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = best_model.fc.in_features
best_model.fc = nn.Linear(num_ftrs, len(categories))
best_model.load_state_dict(torch.load('./model_checkpoint.pth'))
best_model.to(device)

apply_grad_cam('/root/stanfordData4321/stanfordData4321/images3/s-prd-398170393.jpg', best_model, transform, best_model.layer4[1].conv2)
#"/root/stanfordData4321/stanfordData4321/images3/s-prd-398170393.jpg"
# Save the best model
torch.save(best_model.state_dict(), './best_model_checkpoint.pth')
print("Best model saved to './best_model_checkpoint.pth'")
