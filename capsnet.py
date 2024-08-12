import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

# Define the categories (labels) in your dataset
categories = [
    '1-benign-melanocytic nevus', '2-benign-seborrheic keratosis', '3-benign-fibrous papule',
    '4-benign-dermatofibroma', '5-benign-hemangioma', '6-benign-other', '7-malignant-bcc',
    '8-malignant-scc', '9-malignant-sccis', '10-malignant-ak', '11-malignant-melanoma',
    '12-malignant-other', '13-other-melanocytic lesion with possible re-excision',
    '14-other-non-neoplastic/inflammatory/infectious'
]

# Dataset class for handling Excel data
class ExcelImageDataset(Dataset):
    def __init__(self, excel_file, root_dirs, transform=None):
        self.data_frame = pd.read_excel(excel_file)
        if 'Unnamed: 0' in self.data_frame.columns:
            self.data_frame = self.data_frame.drop(columns=['Unnamed: 0'])

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

# Capsule Network implementation
class PrimaryCapsules(nn.Module):
    def __init__(self, num_capsules, in_channels, out_channels, kernel_size, stride):
        super(PrimaryCapsules, self).__init__()
        self.capsules = nn.Conv2d(in_channels, num_capsules * out_channels, kernel_size, stride)
        self.num_capsules = num_capsules
        self.out_channels = out_channels

    def forward(self, x):
        x = self.capsules(x)
        x = x.view(x.size(0), self.num_capsules, -1)
        x = self.squash(x)
        return x

    def squash(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        norm_squared = norm ** 2
        return (norm_squared / (1 + norm_squared)) * (x / norm)

class SecondaryCapsules(nn.Module):
    def __init__(self, num_capsules, num_routes, in_channels, out_channels):
        super(SecondaryCapsules, self).__init__()
        self.num_capsules = num_capsules
        self.num_routes = num_routes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.route_weights = nn.Parameter(
            torch.randn(num_routes, num_capsules, out_channels, in_channels)  # Ensure correct dimensions
        )

    def forward(self, x):
        batch_size = x.size(1)  # Assuming x has shape [num_routes, batch_size, 1, in_channels]
        num_routes = x.size(0)

        print(f"Input tensor shape before view: {x.shape}")
        x = x.view(num_routes, batch_size, -1)  # Flatten the capsule outputs
        print(f"Tensor shape after view: {x.shape}")

        x = x.unsqueeze(2)  # Add new dimension for compatibility
        print(f"Tensor shape after unsqueeze: {x.shape}")

        # Adjust route_weights to match the number of routes
        adjusted_route_weights = self.route_weights[:num_routes, :, :, :]

        # Print shapes for debugging
        print(f"x shape for matmul: {x.shape}")
        print(f"adjusted_route_weights shape for matmul: {adjusted_route_weights.shape}")

        # Permute tensors for correct dimensions
        x = x.permute(0, 1, 3, 2)  # [num_routes, batch_size, in_channels, 1]
        adjusted_route_weights = adjusted_route_weights.permute(0, 1, 3, 2)  # [num_routes, num_capsules, in_channels, out_channels]

        # Print permuted shapes for debugging
        print(f"x shape after permute: {x.shape}")
        print(f"adjusted_route_weights shape after permute: {adjusted_route_weights.shape}")

        try:
            # Perform batch matrix multiplication
            u_hat = torch.matmul(x, adjusted_route_weights)  # Resulting in [num_routes, batch_size, num_capsules, out_channels]
            print(f"u_hat shape: {u_hat.shape}")
        except RuntimeError as e:
            print(f"Matrix multiplication error: {e}")
            raise

        u_hat = u_hat.permute(1, 2, 0, 3)  # Switch back to [batch_size, num_capsules, num_routes, out_channels]
        print(f"u_hat shape after permute: {u_hat.shape}")

        b_ij = torch.zeros(batch_size, self.num_capsules, num_routes, 1).to(x.device)
        for _ in range(3):  # Fixed range syntax
            c_ij = torch.softmax(b_ij, dim=2)
            s_j = (c_ij * u_hat).sum(dim=2, keepdim=True)
            v_j = self.squash(s_j)
            b_ij = b_ij + (u_hat * v_j).sum(dim=-1, keepdim=True)

        return v_j.squeeze(2)

    def squash(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        norm_squared = norm ** 2
        return (norm_squared / (1 + norm_squared)) * (x / norm)



class CapsuleNetwork(nn.Module):
    def __init__(self, num_classes):
        super(CapsuleNetwork, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = PrimaryCapsules(num_capsules=8, in_channels=256, out_channels=32, kernel_size=9, stride=2)
        self.secondary_capsules = SecondaryCapsules(
            num_capsules=num_classes, num_routes=32 * 6 * 6, in_channels=8, out_channels=16
        )

    def forward(self, x):
        x = torch.relu(self.conv_layer(x))
        x = self.primary_capsules(x)
        x = self.secondary_capsules(x)
        return x



# Training function
def train_capsule_network():
    excel_file = '/root/stanfordData4321/stanfordData4321/dataRef/release_midas.xlsx'
    root_dirs = [
    '/root/stanfordData4321/stanfordData4321/images2',
    '/root/stanfordData4321/stanfordData4321/images1',
    '/root/stanfordData4321/stanfordData4321/images3',
    '/root/stanfordData4321/stanfordData4321/images4'
]  # Update this with actual root directories
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = ExcelImageDataset(excel_file=excel_file, root_dirs=root_dirs, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    model = CapsuleNetwork(num_classes=len(categories))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):  # Number of epochs
        model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{10}, Loss: {epoch_loss:.4f}')

    torch.save(model.state_dict(), 'capsule_network.pth')

# Run the training
if __name__ == "__main__":
    train_capsule_network()
