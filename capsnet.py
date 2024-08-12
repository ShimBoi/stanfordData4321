import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split  # Import Dataset
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os

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

class ExcelImageDataset(Dataset):  # Using the Dataset class
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

class AugmentedImageDataset(Dataset):  # Using the Dataset class
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

# Define your CapsNet model
class PrimaryCapsuleLayer(nn.Module):
    def __init__(self, num_capsules, in_channels, out_channels, kernel_size, stride=2):
        super(PrimaryCapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.out_channels = out_channels
        self.capsules = nn.Conv2d(in_channels, num_capsules * out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        x = self.capsules(x)
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_capsules, self.out_channels, -1)
        x = x.permute(0, 3, 1, 2)  # [batch_size, num_routes, num_capsules, out_channels]
        norms = torch.norm(x, dim=-1, keepdim=True)
        squash = (norms**2 / (1 + norms**2)) / (1 + norms**2)
        return squash * x / norms

class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, in_channels, out_channels, num_routes, routing_iterations=3):
        super(CapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.routing_iterations = routing_iterations
        self.route_weights = nn.Parameter(torch.randn(num_capsules, num_routes, in_channels, out_channels))

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1, x.size(-2), x.size(-1))  # Ensure the input is in correct shape

        u_hat = torch.matmul(x.unsqueeze(2), self.route_weights)  # Shape: [batch_size, num_routes, num_capsules, out_channels]
        u_hat = u_hat.squeeze(-2)

        b_ij = torch.zeros_like(u_hat[:, :, :, 0], device=x.device)
        
        for iteration in range(self.routing_iterations):
            c_ij = F.softmax(b_ij, dim=1)
            s_j = (c_ij.unsqueeze(-1) * u_hat).sum(dim=1)
            norms = torch.norm(s_j, dim=-1, keepdim=True)
            v_j = (norms**2 / (1 + norms**2)) * (s_j / norms)
            if iteration < self.routing_iterations - 1:
                b_ij = b_ij + (u_hat * v_j.unsqueeze(1)).sum(dim=-1)
        
        return v_j

class CapsNet(nn.Module):
    def __init__(self, num_classes):
        super(CapsNet, self).__init__()
        self.primary_capsules = PrimaryCapsuleLayer(num_capsules=8, in_channels=3, out_channels=32, kernel_size=9)
        # Calculate the correct number of routes here based on your input image size and the PrimaryCapsuleLayer output.
        num_routes = self._calculate_num_routes()  # Update to calculate the correct num_routes
        self.secondary_capsules = CapsuleLayer(num_capsules=num_classes, in_channels=32, out_channels=16, num_routes=num_routes)
        self.fc = nn.Linear(16 * num_classes, num_classes)

    def _calculate_num_routes(self):
        # Assuming input size is (3, 224, 224)
        with torch.no_grad():
            x = torch.zeros(1, 3, 224, 224)  # Adjust size based on your input
            x = self.primary_capsules.capsules(x)
            num_routes = x.size(2) * x.size(3)
        return num_routes

    def forward(self, x):
        x = self.primary_capsules(x)
        x = self.secondary_capsules(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Initialize dataset and model
root_dirs = [
    '/root/stanfordData4321/stanfordData4321/images2',
    '/root/stanfordData4321/stanfordData4321/images1',
    '/root/stanfordData4321/stanfordData4321/images3',
    '/root/stanfordData4321/stanfordData4321/images4'
]

dataset = ExcelImageDataset('./dataRef/release_midas.xlsx', root_dirs, transform)
augmented_dataset = AugmentedImageDataset(dataset, './augmented_images', transform)

# Split dataset
train_size = int(0.8 * len(augmented_dataset))
test_size = len(augmented_dataset) - train_size
train_dataset, test_dataset = random_split(augmented_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(categories)
model = CapsNet(num_classes=num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
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
