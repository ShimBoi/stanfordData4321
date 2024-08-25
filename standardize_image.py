import os
import cv2
import numpy as np
from tqdm import tqdm

input_folders = ['images1', 'images2', 'images3', 'images4']  
output_base_folder = 'standardized_images'
os.makedirs(output_base_folder, exist_ok=True)

# Define standard size 
standard_size = (700, 700)  

def standardize_image_with_padding(image_path, output_path, size):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image {image_path}. Skipping.")
        return
    # Convert to RGB (if using cv2)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Get current size
    h, w = image.shape[:2]
    # Compute scaling factors
    scale = min(size[0] / h, size[1] / w)
    # Resize while maintaining aspect ratio
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h))

    # Create a new image with the target size and pad the resized image into the center
    pad_image = np.ones((size[0], size[1], 3), dtype=np.uint8) * 255  # Use white padding
    pad_image[(size[0] - new_h) // 2:(size[0] - new_h) // 2 + new_h,
              (size[1] - new_w) // 2:(size[1] - new_w) // 2 + new_w, :] = resized_image

    # Save the padded image
    pad_image = pad_image / 255.0  # Normalize if needed
    cv2.imwrite(output_path, pad_image * 255)  # convert back to 0-255 before saving

for folder in input_folders:
    output_folder = os.path.join(output_base_folder, folder)
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of images in the folder
    image_files = [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in tqdm(image_files, desc=f"Processing {folder}"):
        input_path = os.path.join(folder, image_file)
        output_path = os.path.join(output_folder, image_file)
        standardize_image_with_padding(input_path, output_path, standard_size)

print("Image standardization with padding complete.")
