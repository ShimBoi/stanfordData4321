import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from torchvision import transforms, models
from PIL import Image
import os
import optuna
import cv2
import matplotlib.pyplot as plt

# Define your custom dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_excel(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # Map labels to numeric values
        self.label_map = {label: idx for idx, label in enumerate(self.data_frame['clinical_impression_1'].unique())}
        self.data_frame['label'] = self.data_frame['clinical_impression_1'].map(self.label_map)
        
        # Gather all image paths and labels
        self.image_paths = [os.path.join(root_dir, img_name) for img_name in self.data_frame['midas_file_name']]
        self.labels = self.data_frame['label'].tolist()

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Function to calculate class weights
def compute_class_weights(labels):
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    class_weights = [total_samples / (num_classes * count) for count in class_counts]
    class_weights = [weight / max(class_weights) for weight in class_weights]  # Normalize weights
    return class_weights

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load datasets
dataset = CustomDataset(csv_file='/root/stanfordData4321/stanfordData4321-1/dataRef/release_midas.xlsx', root_dir='/root/stanfordData4321/stanfordData4321-1', transform=transform)

# Get labels from dataset
labels = [label for _, label in dataset]

# Calculate sample weights
class_weights = compute_class_weights(labels)
sample_weights = np.array([class_weights[label] for label in labels])

# Create a sampler for the training dataset
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# Split dataset into training and test sets
indices = np.arange(len(dataset))
train_indices, test_indices = train_test_split(indices, test_size=0.2, stratify=labels)

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

# Create DataLoaders
train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler, num_workers=4)
test_loader = DataLoader(dataset, batch_size=32, sampler=test_sampler, num_workers=4)

# Define model
def create_model():
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, len(set(labels)))  # Adjust output layer
    return model

# Grad-CAM implementation
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = []
        self.activations = []

        def save_gradient(grad):
            self.gradients.append(grad)

        def save_activation(module, input, output):
            self.activations.append(output)

        self.model.layer4[1].register_forward_hook(save_activation)
        self.model.layer4[1].register_backward_hook(save_gradient)
    
    def forward(self, input_tensor):
        self.gradients = []
        self.activations = []
        return self.model(input_tensor)

    def generate_cam(self, input_tensor, target_class):
        self.model.zero_grad()
        output = self.forward(input_tensor)
        class_loss = output[0, target_class]
        class_loss.backward()

        gradients = self.gradients[0].cpu().data.numpy()[0]
        activations = self.activations[0].cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.size(2), input_tensor.size(3)))
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam

# Objective function for Optuna
def objective(trial):
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    momentum = trial.suggest_float('momentum', 0.7, 0.9)

    model = create_model()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training loop
    model.train()
    for epoch in range(2):  # Adjust number of epochs as needed
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total

    # Generate Grad-CAM
    grad_cam = GradCAM(model)
    sample_input, sample_label = next(iter(test_loader))
    sample_input = sample_input[0].unsqueeze(0).to(device)
    target_class = sample_label[0].item()
    
    cam = grad_cam.generate_cam(sample_input, target_class)
    plt.imshow(cam, cmap='jet')
    plt.show()

    return accuracy

# Run Optuna optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print(f"  Params: {trial.params}")

# Train final model with best parameters
best_lr = trial.params['lr']
best_momentum = trial.params['momentum']
final_model = create_model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(final_model.parameters(), lr=best_lr, momentum=best_momentum)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
final_model.to(device)

# Training final model
final_model.train()
for epoch in range(2):  # Adjust number of epochs as needed
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = final_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print("Training complete.")
