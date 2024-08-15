import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from ultralytics import YOLO



# Create directories if they don't exist
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# Load Excel file
excel_file = './dataRef/release_midas.xlsx'
df = pd.read_excel(excel_file)

# Define the label map (you might need to adjust this based on your dataset)
label_map = {
    '7-malignant-bcc': 0,
    '1-benign-melanocytic nevus': 1,
    '6-benign-other': 2,
    '14-other-non-neoplastic/inflammatory/infectious': 3,
    '8-malignant-scc': 4,
    '9-malignant-sccis': 5,
    '10-malignant-ak': 6,
    '3-benign-fibrous papule': 7,
    '4-benign-dermatofibroma': 8,
    '2-benign-seborrheic keratosis': 9,
    '5-benign-hemangioma': 10,
    '11-malignant-melanoma': 11,
    '13-other-melanocytic lesion with possible re-excision (severe/spitz nevus, aimp)': 12,
    '12-malignant-other': 13
}

# Function to convert annotations to YOLO format
def convert_to_yolo_format(row):
    img_name = row['midas_file_name']
    label = row['clinical_impression_1']
    if label not in label_map:
        return None
    label_id = label_map[label]

    # Bounding box info - adjust these columns based on your data
    x_center = (row['bbox_x'] + row['bbox_width'] / 2) / row['img_width']
    y_center = (row['bbox_y'] + row['bbox_height'] / 2) / row['img_height']
    width = row['bbox_width'] / row['img_width']
    height = row['bbox_height'] / row['img_height']
    
    return f"{label_id} {x_center} {y_center} {width} {height}"

# Split data into train and validation
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

def save_annotations(df, split):
    for idx, row in df.iterrows():
        yolo_annotation = convert_to_yolo_format(row)
        if yolo_annotation:
            img_name = row['midas_file_name']
            label_txt_file = os.path.join(labels_dir, img_name.replace('.jpg', '.txt'))
            with open(label_txt_file, 'w') as f:
                f.write(yolo_annotation)
            # Copy image to the correct directory
            shutil.copy(os.path.join('/path/to/original/images', img_name), images_dir)  # Adjust path

# Save annotations for train and validation
save_annotations(train_df, 'train')
save_annotations(val_df, 'val')

# Define the YAML configuration for YOLOv8
dataset_yaml = f'''
path: {data_dir}
train: images/train
val: images/val

nc: {len(label_map)}  # Number of classes
names: {list(label_map.keys())}
'''

# Save the YAML configuration
yaml_path = os.path.join(data_dir, 'dataset.yaml')
with open(yaml_path, 'w') as f:
    f.write(dataset_yaml)

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # or other YOLOv8 model weights

# Train the model
model.train(data=yaml_path, epochs=10, batch=4, imgsz=640, project='yolov8_project')

# Evaluate the model
results = model.val()

# Save the trained model
model.save('yolov8_trained.pt')

print("Training completed and model saved.")
