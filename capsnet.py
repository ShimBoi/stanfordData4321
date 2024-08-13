import torch
import torch.nn as nn
import torch.nn.functional as F

class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, in_channels, out_channels, num_routes):
        super(CapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_routes = num_routes
        
        self.route_weights = nn.Parameter(torch.randn(num_capsules, num_routes, in_channels, out_channels))

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(2)  # Add route dimension
        u_hat = torch.matmul(x, self.route_weights)  # Matrix multiplication
        
        # Dynamic routing logic
        b_ij = torch.zeros(batch_size, self.num_capsules, self.num_routes).to(x.device)
        for _ in range(3):  # Routing iterations
            c_ij = F.softmax(b_ij, dim=2)
            s_j = (c_ij.unsqueeze(3) * u_hat).sum(dim=2)
            v_j = self.squash(s_j)
            if _ < 2:  # Update routing weights
                b_ij = b_ij + (u_hat @ v_j.unsqueeze(3)).squeeze(3)
        
        return v_j
    
    @staticmethod
    def squash(x):
        norm = (x ** 2).sum(dim=-1, keepdim=True)
        scale = norm / (1 + norm) / torch.sqrt(norm + 1e-9)
        return scale * x

class CapsNet(nn.Module):
    def __init__(self, num_classes=15):
        super(CapsNet, self).__init__()
        
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)  # 64x64x3 -> 64x64x8
        self.pool1 = nn.MaxPool2d(2, 2)             # 64x64x8 -> 32x32x8
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1) # 32x32x8 -> 32x32x16
        self.pool2 = nn.MaxPool2d(2, 2)             # 32x32x16 -> 16x16x16

        self.conv3 = nn.Conv2d(16, 32, 3, padding=1) # 16x16x16 -> 16x16x32
        self.pool3 = nn.MaxPool2d(2, 2)              # 16x16x32 -> 8x8x32

        # Primary capsules layer
        self.primary_capsules = CapsuleLayer(num_capsules=32, in_channels=32, out_channels=8, num_routes=8*8)
        
        # Secondary capsules layer (digit capsules)
        self.secondary_capsules = CapsuleLayer(num_capsules=num_classes, in_channels=8, out_channels=16, num_routes=32)
        
        # Decoder network for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(16 * num_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 8 * 8 * 32)
        )
        self.fc = nn.Linear(16 * num_classes, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), 32, 8*8)  # Flatten for primary capsule input
        
        x = self.primary_capsules(x)
        x = self.secondary_capsules(x)
        
        # Flatten and pass through final layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# Instantiate and test the model
net = CapsNet(num_classes=15)
dummy_input = torch.randn(1, 3, 64, 64)  # Batch size of 1, 3 channels, 64x64 image
output = net(dummy_input)
print(output.shape)
