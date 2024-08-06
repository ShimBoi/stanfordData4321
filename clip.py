import pandas as pd
from torchvision import transforms
from PIL import Image
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, excel_file, transform=None):
        self.data = pd.read_excel(excel_file)
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
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertTokenizer, BertModel

class EnhancedCLIPModel(nn.Module):
    def __init__(self):
        super(EnhancedCLIPModel, self).__init__()
        # Vision Encoder
        self.vision_encoder = models.resnet18(pretrained=True)
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
def train(model, dataloader, optimizer, criterion):
    model.train()
    for images, metadata, labels in dataloader:
        optimizer.zero_grad()
        
        # Forward pass
        vision_features, text_features, metadata_features = model(images, labels, metadata)
        
        # Compute similarity and loss
        similarity = model.compute_similarity(vision_features, text_features, metadata_features)
        loss = criterion(similarity, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
