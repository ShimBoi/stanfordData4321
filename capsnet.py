import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os

class CapsuleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_routes, kernel_size, stride, padding):
        super(CapsuleLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * num_routes, kernel_size, stride, padding)
        self.num_routes = num_routes
        self.out_channels = out_channels
    
    def squash(self, x):
        norm = (x ** 2).sum(dim=-1, keepdim=True)
        scale = norm / (1 + norm)
        return scale * (x / torch.sqrt(norm + 1e-8))
    
    def forward(self, x):
        x = self.conv(x)
        batch_size, _, h, w = x.size()
        x = x.view(batch_size, self.num_routes, self.out_channels, h, w)
        x = x.permute(0, 1, 3, 4, 2)  # [batch_size, num_routes, h, w, out_channels]
        x = x.contiguous().view(batch_size, self.num_routes, h * w, self.out_channels)
        x = self.squash(x)
        return x

class PrimaryCapsules(nn.Module):
    def __init__(self):
        super(PrimaryCapsules, self).__init__()
        self.conv = nn.Conv2d(3, 256, kernel_size=9, stride=1)
        self.primary_caps = CapsuleLayer(256, 8, 32, kernel_size=9, stride=2, padding=0)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.primary_caps(x)
        return x

class DigitCapsules(nn.Module):
    def __init__(self, num_classes):
        super(DigitCapsules, self).__init__()
        self.num_classes = num_classes
        self.num_routes = 32  # Number of routes from primary capsules to digit capsules
        self.num_capsules = num_classes
        self.route_weights = nn.Parameter(torch.randn(self.num_routes, self.num_capsules, 8, 16))
        self.num_routing = 3  # Number of routing iterations

    def squash(self, x):
        norm = (x ** 2).sum(dim=-1, keepdim=True)
        scale = norm / (1 + norm)
        return scale * (x / torch.sqrt(norm + 1e-8))

    def forward(self, x):
        batch_size, _, h, w = x.size()
        x = x.view(batch_size, -1, h * w).permute(0, 2, 1)
        x = x.unsqueeze(2).expand(-1, -1, self.num_capsules, -1)
        u_hat = x.matmul(self.route_weights)
        
        b = torch.zeros_like(u_hat[:, :, :, 0])
        for i in range(self.num_routing):
            c = F.softmax(b, dim=2)
            s = (c.unsqueeze(3) * u_hat).sum(dim=1)
            v = self.squash(s)
            if i < self.num_routing - 1:
                b = b + (u_hat * v.unsqueeze(1)).sum(dim=-1)

        return v

class CapsuleNetwork(nn.Module):
    def __init__(self, num_classes):
        super(CapsuleNetwork, self).__init__()
        self.primary_capsules = PrimaryCapsules()
        self.digit_capsules = DigitCapsules(num_classes)

    def forward(self, x):
        x = self.primary_capsules(x)
        x = self.digit_capsules(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, excel_file, root_dirs, transform=None):
        self.df = pd.read_excel(excel_file)
        self.root_dirs = root_dirs
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for index, row in self.df.iterrows():
            image_name = row['midas_path']
            label = row['clinical_impression_1']
            found = False
            for root_dir in root_dirs:
                path = os.path.join(root_dir, image_name)
                if os.path.isfile(path):
                    self.image_paths.append(path)
                    self.labels.append(label)
                    found = True
                    break
            if not found:
                print(f"Warning: Image {image_name} not found in any root directory.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

def train_capsule_network():
    num_classes = 15
    model = CapsuleNetwork(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = CustomDataset(
        excel_file='/root/stanfordData4321/stanfordData4321/dataRef/release_midas.xlsx',
        root_dirs = [
    '/root/stanfordData4321/stanfordData4321/images2',
    '/root/stanfordData4321/stanfordData4321/images1',
    '/root/stanfordData4321/stanfordData4321/images3',
    '/root/stanfordData4321/stanfordData4321/images4'
],  # Replace with actual root directories
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(10):
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch+1}/10], Loss: {loss.item()}")

if __name__ == "__main__":
    train_capsule_network()
