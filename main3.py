import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from torchvision import transforms, models
from PIL import Image
import os
import optuna

# Define your custom dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_excel(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # Map labels to numeric values
        self.label_map = {label: idx for idx, label in enumerate(self.data_frame['clinical_impression_1'].unique())}
        self.data_frame['label'] = self.data_frame['clinical_impression_1'].map(self.label_map)

    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 2])  # Adjust the column index for image paths
        image = Image.open(img_name).convert('RGB')
        label = self.data_frame.iloc[idx, -1]  # Label column index
        
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
dataset = CustomDataset(csv_file='/path/to/your/excel_file.xlsx', root_dir='/path/to/your/images', transform=transform)

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
def create_model(lr, momentum):
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, len(set(labels)))  # Adjust output layer
    return model

# Objective function for Optuna
def objective(trial):
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    momentum = trial.suggest_float('momentum', 0.7, 0.9)

    model = create_model(lr, momentum)
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
final_model = create_model(best_lr, best_momentum)
# Train final model with best parameters (code similar to above training loop)

print("Training complete.")
