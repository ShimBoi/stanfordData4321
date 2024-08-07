import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torch import nn, optim
from torchvision import models
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2

# Verify CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define augmentation transformation
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(),
    transforms.RandomResizedCrop(224)
])

# Define the ExcelImageDataset class
class ExcelImageDataset(Dataset):
    def __init__(self, excel_file_path, root_dirs, transform=None):
        self.excel_file_path = excel_file_path
        self.root_dirs = root_dirs
        self.transform = transform
        self.image_paths, self.labels = self._load_data_from_excel()

    def _load_data_from_excel(self):
        df = pd.read_excel(self.excel_file_path)
        image_paths = []
        labels = []
        for _, row in df.iterrows():
            for root_dir in self.root_dirs:
                image_path = os.path.join(root_dir, row['image_name'])
                if os.path.exists(image_path):
                    image_paths.append(image_path)
                    labels.append(row['label'])
                    break
        print(f"Total valid paths found: {len(image_paths)}")
        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label)

# Function to save augmented images
def save_augmented_images(dataset, output_dir, num_augmentations=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, (img, label) in enumerate(dataset):
        label_dir = os.path.join(output_dir, str(label.item()))
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        
        for i in range(num_augmentations):
            augmented_img = augmentation_transforms(img)
            augmented_img = transforms.ToPILImage()(augmented_img)  # Convert back to PIL Image
            augmented_img.save(os.path.join(label_dir, f"{idx}_aug_{i}.png"))

# Load the dataset
excel_file_path = '/content/drive/MyDrive/midasmultimodalimagedatasetforaibasedskincancer/release_midas.xlsx'
root_dirs = [
    '/root/stanfordData4321/stanfordData4321/images2',
    '/root/stanfordData4321/stanfordData4321/images1',
    '/root/stanfordData4321/stanfordData4321/images3',
    '/root/stanfordData4321/stanfordData4321/images4'
]

dataset = ExcelImageDataset(excel_file_path, root_dirs, transform)
output_dir = './augmented_images'
save_augmented_images(dataset, output_dir, num_augmentations=5)

# Load the augmented dataset
augmented_dataset = AugmentedImageDataset(output_dir, transform)

# Split the dataset
train_size = int(0.8 * len(augmented_dataset))
test_size = len(augmented_dataset) - train_size
train_dataset, test_dataset = random_split(augmented_dataset, [train_size, test_size])

print(f"Train size: {train_size}, Test size: {test_size}")
print(f"Train dataset length: {len(train_dataset)}, Test dataset length: {len(test_dataset)}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Define the model
weights = models.ResNet18_Weights.DEFAULT
net = models.resnet18(weights=weights)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, len(set(augmented_dataset.labels)))

net.to(device)
print(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

# Training loop
for epoch in range(15):
    running_loss = 0.0
    print(epoch)
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
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}")
            running_loss = 0.0

print("Finished Training")

# Save the trained model
PATH = './cifar_net.pth'
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

print(f"Accuracy of the network: {100 * correct / total:.2f} %")

# Grad-CAM explanation
def get_grad_cam_explanation(vision_model, image, target_layer):
    cam = GradCAM(model=vision_model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=image.unsqueeze(0))
    image = image.permute(1, 2, 0).cpu().numpy()
    cam_image = show_cam_on_image(image, grayscale_cam[0, :], use_rgb=True)
    return cam_image

# Example usage
target_layer = net.layer4[-1].conv2  # Adjust target layer
sample_image, _ = augmented_dataset[0]
sample_image = sample_image.to(device)
cam_image = get_grad_cam_explanation(net, sample_image, target_layer)

print("Grad-CAM Image Generated")

# Function to visualize the Grad-CAM image
def visualize_grad_cam(cam_image):
    plt.imshow(cam_image)
    plt.axis('off')
    plt.show()

# Function to save the Grad-CAM image
def save_grad_cam(cam_image, filename='grad_cam_output.png'):
    plt.imsave(filename, cam_image)

visualize_grad_cam(cam_image)
save_grad_cam(cam_image, 'grad_cam_image.png')
print("Grad-CAM Image Generated and Saved")
