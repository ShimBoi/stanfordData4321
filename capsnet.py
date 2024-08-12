import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

# Capsule Network components

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        return F.relu(self.conv(x))

class PrimaryCapsules(nn.Module):
    def __init__(self, num_capsules, in_channels, out_channels, kernel_size, stride):
        super(PrimaryCapsules, self).__init__()
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, stride) 
            for _ in range(num_capsules)
        ])
        self.out_channels = out_channels

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        batch_size = x.size(0)
        u = u.view(batch_size, -1, self.out_channels)
        return u

class DigitCapsules(nn.Module):
    def __init__(self, num_capsules, num_routes, in_channels, out_channels):
        super(DigitCapsules, self).__init__()
        self.num_routes = num_routes
        self.num_capsules = num_capsules
        self.weights = nn.Parameter(torch.randn(num_capsules, num_routes, in_channels, out_channels))

    def forward(self, x):
        u = torch.matmul(x[:, None, :, :, None], self.weights)
        u = u.squeeze(-1).permute(0, 3, 1, 2)
        return self.squash(u.sum(dim=2))

    def squash(self, x, dim=-1):
        norm = torch.norm(x, dim=dim, keepdim=True)
        scale = (norm ** 2) / (1 + norm ** 2)
        return scale * (x / norm)

class CapsuleNetwork(nn.Module):
    def __init__(self, num_classes):
        super(CapsuleNetwork, self).__init__()
        self.conv1 = ConvLayer(in_channels=3, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = PrimaryCapsules(num_capsules=8, in_channels=256, out_channels=32, kernel_size=9, stride=2)
        self.digit_capsules = DigitCapsules(num_capsules=num_classes, num_routes=32*6*6, in_channels=8, out_channels=16)

    def forward(self, x):
        x = self.conv1(x)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x)
        return x

# Custom Dataset class

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

# Load dataset

categories = ['7-malignant-bcc', '1-benign-melanocytic nevus', '6-benign-other',
              '14-other-non-neoplastic/inflammatory/infectious', '8-malignant-scc',
              '9-malignant-sccis', '10-malignant-ak', '3-benign-fibrous papule',
              '4-benign-dermatofibroma', '2-benign-seborrheic keratosis',
              '5-benign-hemangioma', '11-malignant-melanoma',
              '13-other-melanocytic lesion with possible re-excision (severe/spitz nevus, aimp)',
              '12-malignant-other']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

root_dirs = [
    '/root/stanfordData4321/stanfordData4321/images2',
    '/root/stanfordData4321/stanfordData4321/images1',
    '/root/stanfordData4321/stanfordData4321/images3',
    '/root/stanfordData4321/stanfordData4321/images4'
]

dataset = ExcelImageDataset('./dataRef/release_midas.xlsx', root_dirs, transform)

# Split dataset

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Model training and evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CapsuleNetwork(num_classes=len(categories))
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):  # Adjust epoch count as needed
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch+1} loss: {running_loss/len(train_loader):.4f}')

print('Finished Training')

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test dataset: {100 * correct / total:.2f}%')
