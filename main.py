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
import torchvision
from transformers import GPT2Tokenizer, GPT2Model
import cv2
import os

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the Excel file
excel_file_path = './dataRef/release_midas.xlsx'
if not os.path.exists(excel_file_path):
    raise FileNotFoundError(f"{excel_file_path} does not exist. Please check the path.")

df = pd.read_excel(excel_file_path)

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

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

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
            img_name = row['midas_file_name']
            label = row['clinical_impression_1']

            # Check in both root directories
            found = False
            for root_dir in self.root_dirs:
                full_img_path = os.path.join(root_dir, img_name)
                if os.path.isfile(full_img_path):
                    if label in self.label_map:
                        valid_paths.append((full_img_path, label))
                    else:
                        print(f"Warning: Label '{label}' not in label_map for image {full_img_path}.")
                    found = True
                    break
            
            if not found:
                print(f"Warning: {img_name} not found in any of the root directories.")

        print(f"Total valid images found: {len(valid_paths)}")
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


# Example usage with multiple root directories
root_dirs = [
    '/root/stanfordData4321/stanfordData4321/images1',
    '/root/stanfordData123/stanfordData123/images2',
    '/root/stanfordData123/stanfordData123/images3',
    '/root/stanfordData123/stanfordData123/images4'
]
data = ExcelImageDataset(excel_file=excel_file_path, root_dirs=root_dirs, transform=transform)

# 8:2 split for train and test
train_size = int(0.80 * len(data))
test_size = len(data) - train_size
train_data, test_data = random_split(data, [train_size, test_size])

# DataLoaders for training and testing data
batch_size = 16
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Example of showing a batch of images
dataiter = iter(train_loader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images.cpu()))  # Move to CPU for plotting

# Load a ResNet model and modify the final layer
net = models.resnet18(weights='IMAGENET1K_V1')  # Updated to use 'weights'
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, len(categories))
net.to(device)  # Move model to the device

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

# Create a directory to save model weights if it doesn't exist
os.makedirs('saved_models', exist_ok=True)

# Load from a checkpoint if available
start_epoch = 0
checkpoint_path = 'saved_models/resnet_epoch_25.pth'  # Change this to the latest checkpoint
if os.path.exists(checkpoint_path):
    net.load_state_dict(torch.load(checkpoint_path))
    start_epoch = int(checkpoint_path.split('_')[-1].split('.')[0])

# Initialize GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_model = GPT2Model.from_pretrained("gpt2")

# Set padding token
tokenizer.pad_token = tokenizer.eos_token

# Training loop
num_epochs = 6
for epoch in range(start_epoch, num_epochs):
    print("Epoch", epoch + 1)
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to device

        optimizer.zero_grad()

        # Process images through ResNet
        outputs = net(inputs)

        # Create dummy text inputs for GPT-2 with placeholder text
        dummy_texts = ["Sample text"] * inputs.size(0)  # Use non-empty placeholder text
        text_inputs = tokenizer(dummy_texts, return_tensors="pt", padding=True, truncation=True)

        # Debugging: Print text inputs shapes and content
        print(f"text_inputs input_ids shape: {text_inputs['input_ids'].shape}")

        # Check if text_inputs are valid
        if text_inputs['input_ids'].size(1) == 0:
            raise ValueError("Text inputs are empty or invalid")

        # Move text inputs to the same device as the model
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        # Process text inputs through GPT-2
        try:
            gpt_outputs = gpt_model(**text_inputs).last_hidden_state
        except Exception as e:
            print(f"Error during GPT-2 processing: {e}")
            continue

        # Ensure combined_outputs tensor is correctly formed
        if outputs.size(0) != gpt_outputs.size(0):
            raise ValueError("Mismatch in batch sizes between image and text outputs")

        # Combine the outputs

        # Forward pass through ResNet
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}")
            running_loss = 0.0

print("Finished Training")

# Save the final model
torch.save(net.state_dict(), "saved_models/resnet_final.pth")

# Evaluate the model
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

print(f"Accuracy of the network: {100 * correct / total} %")

# Grad-CAM implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handlers = []
        self._register_hooks()

    def _register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        self.hook_handlers.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handlers.append(self.target_layer.register_backward_hook(backward_hook))

    def generate(self, input_image, class_idx):
        self.model.zero_grad()
        output = self.model(input_image)
        one_hot = torch.zeros((1, output.size()[-1]), device=output.device)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        gradients = self.gradients
        activations = self.activations

        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        grad_cam = torch.sum(weights * activations, dim=1, keepdim=True)
        grad_cam = torch.clamp(grad_cam, min=0)
        grad_cam = nn.functional.interpolate(grad_cam, size=input_image.shape[2:], mode='bilinear', align_corners=False)

        return grad_cam.squeeze().detach()

    def __del__(self):
        for handler in self.hook_handlers:
            handler.remove()

# Usage of Grad-CAM
grad_cam = GradCAM(model=net, target_layer=net.layer4[1].conv2)

# Assuming `inputs` is a batch of test images and `predicted` are the predicted labels
inputs, labels = next(iter(test_loader))
inputs, labels = inputs.to(device), labels.to(device)
outputs = net(inputs)
_, predicted = torch.max(outputs.data, 1)

# Generate and visualize Grad-CAM heatmap for the first image in the batch
heatmap = grad_cam.generate(inputs[0].unsqueeze(0), predicted[0].item())
plt.imshow(heatmap.cpu().numpy(), cmap='jet')
plt.show()
