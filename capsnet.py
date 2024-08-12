import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import optuna
from collections import Counter
import torch.nn.functional as F

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

# Count images before augmentation
pre_augmentation_counts = count_images_per_label(dataset)
print("Image counts before augmentation:")
for label, count in pre_augmentation_counts.items():
    print(f"{label}: {count}")

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

# Count images after augmentation
post_augmentation_counts = count_images_per_label(augmented_dataset)
print("Image counts after augmentation:")
for label, count in post_augmentation_counts.items():
    print(f"{label}: {count}")

# Split dataset
train_size = int(0.8 * len(augmented_dataset))
test_size = len(augmented_dataset) - train_size
train_dataset, test_dataset = random_split(augmented_dataset, [train_size, test_size])

print(f"Train dataset length: {len(train_dataset)}, Test dataset length: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Define Capsule layers and Capsule Network

class PrimaryCapsules(nn.Module):
    def __init__(self, num_capsules, in_channels, out_channels, kernel_size, stride):
        super(PrimaryCapsules, self).__init__()
        self.capsules = nn.Conv2d(in_channels, num_capsules * out_channels, kernel_size=kernel_size, stride=stride)
        self.num_capsules = num_capsules
        self.out_channels = out_channels

    def forward(self, x):
        out = self.capsules(x)
        N, C, H, W = out.size()
        out = out.view(N, self.num_capsules, self.out_channels, H, W)
        out = out.permute(0, 1, 3, 4, 2).contiguous()  # (N, num_capsules, H, W, out_channels)
        out = out.view(out.size(0), -1, out.size(-1))
        return self.squash(out)

    def squash(self, x, dim=-1):
        norm = torch.norm(x, p=2, dim=dim, keepdim=True)
        scale = (norm ** 2) / (1 + norm ** 2)
        return scale * x / (norm + 1e-8)

class DigitCapsules(nn.Module):
    def __init__(self, num_capsules, num_routes, in_channels, out_channels):
        super(DigitCapsules, self).__init__()
        self.num_routes = num_routes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weights = nn.Parameter(torch.randn(1, num_routes, num_capsules, in_channels, out_channels))

    def forward(self, x):
        u = torch.matmul(x[:, None, :, :, None], self.weights)
        u = u.squeeze(-1)
        b = torch.zeros_like(u[:, :, :, 0])

        for _ in range(3):  # dynamic routing iterations
            c = F.softmax(b, dim=-1)
            s = (c[:, :, :, None] * u).sum(dim=2)
            v = self.squash(s)
            b = b + torch.matmul(u, v.unsqueeze(-1)).squeeze(-1)
        
        return v

    def squash(self, x, dim=-1):
        norm = torch.norm(x, p=2, dim=dim, keepdim=True)
        scale = (norm ** 2) / (1 + norm ** 2)
        return scale * x / (norm + 1e-8)

class CapsNet(nn.Module):
    def __init__(self, num_classes, img_size, in_channels=3):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=9, stride=1)
        self.primary_capsules = PrimaryCapsules(num_capsules=8, in_channels=256, out_channels=32, kernel_size=9, stride=2)
        self.digit_capsules = DigitCapsules(num_capsules=num_classes, num_routes=32 * 6 * 6, in_channels=8, out_channels=16)
        self.decoder = nn.Sequential(
            nn.Linear(16 * num_classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, img_size * img_size * in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x)
        x = x.view(x.size(0), -1)
        reconstructions = self.decoder(x)
        reconstructions = reconstructions.view(x.size(0), 3, img_size, img_size)
        return x.norm(dim=-1), reconstructions

# Define Capsule Loss

class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()

    def forward(self, output, labels, images, reconstructions):
        margin_loss = self.margin_loss(output, labels)
        reconstruction_loss = F.mse_loss(reconstructions, images)
        return margin_loss + 0.0005 * reconstruction_loss

    def margin_loss(self, output, labels):
        left = F.relu(0.9 - output).view(output.size(0), -1)
        right = F.relu(output - 0.1).view(output.size(0), -1)
        loss = labels * left + 0.5 * (1.0 - labels) * right
        return loss.sum(dim=1).mean()

# Initialize CapsNet model

num_classes = len(categories)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CapsNet(num_classes=num_classes, img_size=img_size).to(device)

criterion = CapsuleLoss()
optimizer = optim.Adam(model.parameters())

# Training the CapsNet

epochs = 20

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs, reconstructions = model(inputs)
        loss = criterion(outputs, labels, inputs, reconstructions)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:  # print every 200 mini-batches
            print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 200:.4f}')
            running_loss = 0.0

print('Finished Training')

# Save the model
torch.save(model.state_dict(), 'capsnet.pth')

# Evaluation with Grad-CAM

def evaluate_and_visualize(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs, _ = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy:.2f}%')

    # Grad-CAM visualization
    cam = GradCAM(model=model, target_layers=[model.digit_capsules], use_cuda=device.type == 'cuda')
    
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        grayscale_cam = cam(input_tensor=images)
        for i in range(images.size(0)):
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            cam_image = show_cam_on_image(img, grayscale_cam[i], use_rgb=True)
            plt.imshow(cam_image)
            plt.show()
        break  # Display Grad-CAM for the first batch only

# Evaluate and visualize the results
evaluate_and_visualize(model, test_loader, device)
