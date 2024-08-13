--upgrade setuptools wheel
pandas
--no-deps google-colab
google-auth==1.4.2
ipykernel==4.6.1
ipython==5.5.0
notebook==5.2.2
six==1.12.0
scikit-learn
pytorch
git+https://github.com/openai/CLIP.git
torch 
torchvision 
pillow 
transformers

import torch
import clip
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

# Load CLIP model and preprocessing
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model, preprocess = clip.load('ViT-B/32', device=device)

# Load Excel file
excel_path =  '/content/drive/MyDrive/midasmultimodalimagedatasetforaibasedskincancer/release_midas.xlsx'
  # Path to your uploaded Excel file
data = pd.read_excel(excel_path)

# Example columns: 'image_name', 'metadata1', 'metadata2', ...
image_names = data['midas_file_name'].tolist()
metadata = data.drop(columns=['midas_file_name', 'midas_record_id', 'clinical_impression_2', 'clinical_impression_3']) #make sure its strictly string or int

# Convert metadata to textual descriptions
metadata_texts = metadata.apply(lambda row: ' '.join(row.astype(str)), axis=1).tolist()

# Prepare images and their corresponding metadata
images = []
metadata_texts_for_images = []
#/content/drive/MyDrive/midasmultimodalimagedatasetforaibasedskincancer/s-prd-396524710.jpg

for image_name, meta_text in zip(image_names, metadata_texts):
    image_path = f'/content/drive/MyDrive/midasmultimodalimagedatasetforaibasedskincancer/{image_name}'  # Path to your uploaded images folder
    if os.path.exists(image_path):
        image = Image.open(image_path)
        images.append(image)
        metadata_texts_for_images.append(meta_text)
    else:
        print(f"File not found: {image_path}")

if not images:
    raise ValueError("No images were loaded. Check the file paths and ensure images exist.")

print(f"Loaded {len(images)} images and corresponding metadata texts.")


# Define function to compute CLIP embeddings for image and text
def get_clip_embeddings(image, text_descriptions):
    # Prepare the image
    image = preprocess(image).unsqueeze(0).to(device)

    # Tokenize and truncate the text descriptions to the maximum context length of 77 tokens
    truncated_texts = [text[:75] + '...' if len(text) > 75 else text for text in text_descriptions]
    text_inputs = clip.tokenize(truncated_texts).to(device)

    # Compute embeddings
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)

    return image_features, text_features

# Compute embeddings for metadata texts
print("Computing embeddings for metadata texts...")
_, text_features = get_clip_embeddings(images[0], metadata_texts_for_images)
print(1)
# Collect image embeddings
image_features_list = []
for image in images:
    image_features, _ = get_clip_embeddings(image, metadata_texts_for_images)
    image_features_list.append(image_features.cpu().numpy())

print("Computed image embeddings.")

# Convert list to numpy array
image_features_array = np.vstack(image_features_list)

# Convert text features to numpy array
text_features_array = text_features.cpu().numpy()

# Compute cosine similarity between image and text features
print("Computing cosine similarity...")
similarity_matrix = image_features_array @ text_features_array.T
predicted_labels = np.argmax(similarity_matrix, axis=1)

# Define label encoder and transform labels for classification
label_encoder = LabelEncoder()
label_encoder.fit(metadata_texts_for_images)  # Fit on your metadata descriptions
predicted_labels_names = label_encoder.inverse_transform(predicted_labels)

# Print predicted labels for each image
print("Printing predicted labels for each image:")
for image_name, label in zip(image_names, predicted_labels_names):
    print(f"Image: {image_name} -> Predicted Label: {label}")
