import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from PIL import Image
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
import numpy as np

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

# Compute class weights globally
labels = [label for _, label in combined_image_paths]
classes = np.array(list(label_map.values()))  # Convert to NumPy array
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# Create sampler
weights = [class_weights[label] for _, label in combined_image_paths]
sampler = WeightedRandomSampler(weights, len(weights))

# Create DataLoader with sampler
train_loader = DataLoader(dataset, batch_size=4, sampler=sampler)

# Split dataset into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

print(f"Train size: {train_size}, Test size: {test_size}")

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

print(f"Train dataset length: {len(train_dataset)}, Test dataset length: {len(test_dataset)}")

# Define the objective function for Optuna
def objective(trial: Trial):
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters to tune
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    momentum = trial.suggest_float("momentum", 0.5, 0.9)
    
    # Define loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Create DataLoader with sampler
    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Print dataset lengths
    print(f"Length of train_dataset: {len(train_dataset)}")
    print(f"Length of test_dataset: {len(test_dataset)}")
    
    # Load pre-trained model and modify the final layer
    weights = models.ResNet18_Weights.DEFAULT
    net = models.resnet18(weights=weights)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, len(categories))

    # Move the model to GPU
    net.to(device)
    
    # Define optimizer with hyperparameters
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    
    # Training loop
    for epoch in range(15):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            try:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            except Exception as e:
                print(f"Error during training loop: {e}")
    
    # Validation loop
    net.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            try:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            except Exception as e:
                print(f"Error during validation loop: {e}")
    
    accuracy = 100 * correct / total
    val_loss /= len(test_loader)
    
    return val_loss  # or use -accuracy to maximize accuracy



# Create a study and optimize the objective function
study = optuna.create_study(direction="minimize", sampler=TPESampler())
study.optimize(objective, n_trials=1)  # Adjust the number of trials

# Print the best hyperparameters
print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print(f"  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Load pre-trained model with the best hyperparameters and train the final model
best_lr = trial.params['lr']
best_momentum = trial.params['momentum']

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

# Define loss function and optimizer with best hyperparameters
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.SGD(net.parameters(), lr=best_lr, momentum=best_momentum)

# Training loop
for epoch in range(15):
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
    
    # Validation loop
    net.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    val_loss /= len(test_loader)
    
    print(f'Epoch [{epoch+1}/15] - Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
