import pandas as pd
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, excel_file, transform=None):
        self.data = pd.read_excel(excel_file, engine='openpyxl')  # Specify engine manually
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image
        img_path = self.data.iloc[idx]['midas_path']
        image = Image.open(img_path).convert('RGB')

        # Apply transformations to image
        if self.transform:
            image = self.transform(image)

        # Load metadata
        metadata = self.data.iloc[idx][[
            'midas_iscontrol', 'midas_distance', 'midas_location', 
            'midas_pathreport', 'midas_gender', 'midas_age', 
            'midas_fitzpatrick', 'midas_melanoma', 'midas_ethnicity', 
            'midas_race', 'length_(mm)', 'width_(mm)'
        ]].to_dict()

        # Load label
        label = self.data.iloc[idx]['clinical_impression_1']

        return image, metadata, label

class EnhancedCLIPModel(nn.Module):
    def __init__(self):
        super(EnhancedCLIPModel, self).__init__()
        # Vision Encoder
        self.vision_encoder = models.resnet18(weights='IMAGENET1K_V1')
        self.vision_encoder.fc = nn.Identity()  # Remove the classification layer
        self.vision_projection = nn.Linear(512, 512)  # Project to common space

        # Text Encoder
        self.text_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.text_projection = nn.Linear(768, 512)  # Project to common space

        # Metadata Encoder
        self.metadata_fc = nn.Linear(12, 512)  # Project metadata to common space

    def forward(self, images, texts, metadata):
        # Vision Encoding
        vision_features = self.vision_encoder(images)
        vision_features = self.vision_projection(vision_features)

        # Text Encoding
        inputs = self.text_tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        text_features = self.text_encoder(**inputs).pooler_output
        text_features = self.text_projection(text_features)

        # Metadata Encoding
        metadata_features = torch.tensor(metadata, dtype=torch.float32)
        metadata_features = self.metadata_fc(metadata_features)

        return vision_features, text_features, metadata_features

    def compute_similarity(self, vision_features, text_features, metadata_features):
        vision_features = nn.functional.normalize(vision_features, p=2, dim=-1)
        text_features = nn.functional.normalize(text_features, p=2, dim=-1)
        metadata_features = nn.functional.normalize(metadata_features, p=2, dim=-1)

        # Concatenate vision, text, and metadata features
        combined_features = torch.cat((vision_features, text_features, metadata_features), dim=-1)
        similarity = torch.matmul(combined_features, combined_features.T)
        return similarity

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    for images, metadata, labels in dataloader:
        images = images.to(device)
        metadata = torch.tensor(metadata, dtype=torch.float32).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        vision_features, text_features, metadata_features = model(images, labels, metadata)

        # Compute similarity and loss
        similarity = model.compute_similarity(vision_features, text_features, metadata_features)
        loss = criterion(similarity, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        print(f'Loss: {loss.item()}')

def evaluate(model, dataloader, device):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, metadata, labels in dataloader:
            images = images.to(device)
            metadata = torch.tensor(metadata, dtype=torch.float32).to(device)
            labels = labels.to(device)

            # Forward pass
            vision_features, text_features, metadata_features = model(images, labels, metadata)

            # Compute similarity and get predictions
            similarity = model.compute_similarity(vision_features, text_features, metadata_features)
            _, predictions = torch.max(similarity, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions, target_names=dataloader.dataset.classes)

    return accuracy, report

def visualize_predictions(model, dataloader, device, num_samples=5):
    model.eval()
    with torch.no_grad():
        for images, metadata, labels in dataloader:
            images = images.to(device)
            metadata = torch.tensor(metadata, dtype=torch.float32).to(device)
            labels = labels.to(device)

            # Forward pass
            vision_features, text_features, metadata_features = model(images, labels, metadata)

            # Compute similarity and get predictions
            similarity = model.compute_similarity(vision_features, text_features, metadata_features)
            _, predictions = torch.max(similarity, dim=1)

            # Display results
            for i in range(num_samples):
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(images[i].cpu().permute(1, 2, 0))
                plt.title(f'True: {labels[i].item()}, Pred: {predictions[i].item()}')
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.bar(['True', 'Pred'], [labels[i].item(), predictions[i].item()])
                plt.title(f'Sample {i}')
                plt.show()

            break  # Remove this break to visualize more samples

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transformations
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize dataset and dataloader
dataset = CustomDataset(excel_file='/root/stanfordData4321/stanfordData4321/dataRef/release_midas.xlsx', transform=data_transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model = EnhancedCLIPModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()  # or your chosen loss function

# Run training
train(model, dataloader, optimizer, criterion, device)

# Assuming you have a separate validation dataset
val_dataset = CustomDataset(excel_file='/content/drive/MyDrive/midasmultimodalimagedatasetforaibasedskincancer/val_midas.xlsx', transform=data_transform)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Run evaluation
accuracy, report = evaluate(model, val_dataloader, device)

print(f'Accuracy: {accuracy:.4f}')
print('Classification Report:')
print(report)

# Visualize predictions
visualize_predictions(model, val_dataloader, device)
