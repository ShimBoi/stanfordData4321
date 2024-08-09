import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import optuna
import numpy as np

# Load the Excel file
excel_file_path = './dataRef/release_midas.xlsx'
if not os.path.exists(excel_file_path):
    raise FileNotFoundError(f"{excel_file_path} does not exist. Please check the path.")

df = pd.read_excel(excel_file_path)
print("Excel file loaded. First few rows:")
print(df.head())

# Define categories and image size
categories = [
    '7-malignant-bcc', '1-benign-melanocytic nevus', '6-benign-other',
    '14-other-non-neoplastic/inflammatory/infectious', '8-malignant-scc',
    '9-malignant-sccis', '10-malignant-ak', '3-benign-fibrous papule',
    '4-benign-dermatofibroma', '2-benign-seborrheic keratosis',
    '5-benign-hemangioma', '11-malignant-melanoma',
    '13-other-melanocytic lesion with possible re-excision (severe/spitz nevus, aimp)',
    '12-malignant-other'
]
img_size = 224

# Compose the transformation pipeline
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label

# Load image paths from the original directories
def load_original_image_paths(df, root_dirs, label_map):
    image_paths = []
    for idx, row in df.iterrows():
        img_name = row['midas_file_name']
        label = row['clinical_impression_1']
        if label not in label_map:
            continue
        for root_dir in root_dirs:
            img_path = os.path.join(root_dir, img_name)
            if os.path.isfile(img_path):
                image_paths.append((img_path, label_map[label]))
                break
    return image_paths

# Load augmented images from the directory
def load_augmented_image_paths(augmented_dir, label_map):
    image_paths = []
    for root, _, files in os.walk(augmented_dir):
        for file in files:
            if file.endswith('.png'):
                label = int(os.path.basename(root))
                img_path = os.path.join(root, file)
                if label in label_map.values():
                    image_paths.append((img_path, label))
    return image_paths

# Define the root directories for original images and augmented images
root_dirs = [
    '/root/stanfordData4321/stanfordData4321/images2',
    '/root/stanfordData4321/stanfordData4321/images1',
    '/root/stanfordData4321/stanfordData4321/images3',
    '/root/stanfordData4321/stanfordData4321/images4'
]
augmented_dir = './augmented_images'  # Directory containing the augmented images

label_map = {label: idx for idx, label in enumerate(categories)}

# Load original and augmented image paths
original_image_paths = load_original_image_paths(df, root_dirs, label_map)
augmented_image_paths = load_augmented_image_paths(augmented_dir, label_map)

# Combine original and augmented image paths
combined_image_paths = original_image_paths + augmented_image_paths

print(f"Total original images: {len(original_image_paths)}")
print(f"Total augmented images: {len(augmented_image_paths)}")
print(f"Total images used in dataset: {len(combined_image_paths)}")

# Create dataset
dataset = ImageDataset(combined_image_paths, transform)
print(f"Dataset length: {len(dataset)}")

# Calculate class weights
class_counts = np.array([961, 703, 454, 376, 224, 165, 152, 124, 83, 49, 40, 11, 9, 9])
total_count = np.sum(class_counts)
class_weights = total_count / (len(class_counts) * class_counts)
print("Class Weights:", class_weights)

# Define the device before using it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert class weights to a tensor and move it to the GPU if available
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Split dataset into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

print(f"Train size: {train_size}, Test size: {test_size}")

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

print(f"Train dataset length: {len(train_dataset)}, Test dataset length: {len(test_dataset)}")

# Compute sample weights for WeightedRandomSampler
sample_weights = [class_weights[label] for _, label in combined_image_paths]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# Use the sampler in your DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Load pre-trained model and modify the final layer
weights = models.ResNet18_Weights.DEFAULT
net = models.resnet18(weights=weights)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, len(categories))

# Move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
print(device)

# Define the objective function for Optuna
def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-1)
    momentum = trial.suggest_uniform('momentum', 0.7, 0.9)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    # Training loop
    for epoch in range(5):  # Loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # Get the inputs and move them to GPU
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()

    # Validation accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Create Optuna study and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=2)

# Get the best trial
best_trial = study.best_trial
print(f"Best trial value (accuracy): {best_trial.value}")
print("Best hyperparameters: ", best_trial.params)

# Use the best hyperparameters to re-train the model
lr = best_trial.params['lr']
momentum = best_trial.params['momentum']
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Training loop with best parameters
for epoch in range(5):
    running_loss = 0.0
    print(f"Epoch {epoch+1}")
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

print("Finished Training")

# Save the trained model
PATH = './resnet18_model.pth'
torch.save(net.state_dict(), PATH)

# Test the model
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the network: {100 * correct / total:.2f}%")

# Grad-CAM visualization
def visualize_gradcam(img_path, class_idx, model, transform):
    image = Image.open(img_path).convert("RGB")
    input_image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_image)
        pred_class = output.argmax(dim=1).item()

    # Initialize GradCAM
    grad_cam = GradCAM(model=model, target_layers=[model.layer4[-1]])

    # Generate GradCAM
    grayscale_cam = grad_cam(input_tensor=input_image, target_category=class_idx)
    grayscale_cam = grayscale_cam[0, :]

    # Convert to numpy and show
    image = np.array(image) / 255.0
    cam_image = show_cam_on_image(image, grayscale_cam, use_rgb=True)
    plt.imshow(cam_image)
    plt.title(f'Grad-CAM for class: {categories[class_idx]}')
    plt.axis('off')
    plt.show()

# Example usage of Grad-CAM
test_image_path = '/path/to/sample_image.png'  # Change to your test image path
test_class_idx = 1  # Change to your test class index
visualize_gradcam(test_image_path, test_class_idx, net, transform)
