import pandas as pd
import os
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

class ExcelImageDataset:
    def __init__(self, excel_file, image_dirs, transform=None):
        self.data = pd.read_excel(excel_file)
        self.image_dirs = image_dirs
        self.transform = transform
        self.labels = self.data['clinical_impression_1'].values  # Assuming this is the label column
        self.image_paths = self.data['midas_path'].values  # Assuming this column has the image paths
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # Search in multiple directories
        for root in self.image_dirs:
            full_path = os.path.join(root, image_path)
            if os.path.exists(full_path):
                image = load_img(full_path)
                break
        else:
            raise FileNotFoundError(f"Image {image_path} not found in any of the directories.")
        
        image = img_to_array(image)
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

    def get_labels(self):
        return self.labels
from keras import layers, models
from keras.utils import to_categorical

# Assuming your images are resized to 128x128x3 and you have 15 classes
input_shape = (128, 128, 3)
n_classes = 15

# Define the model
inputs = layers.Input(shape=input_shape)

# Primary Capsule Layer
x = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu')(inputs)
x = PrimaryCap(x, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

# Capsule Layer
capsule = CapsuleLayer(num_capsule=n_classes, dim_capsule=16, routings=3, name='capsule')(x)

# Length Layer for Output
out_caps = Length(name='out_caps')(capsule)

# Decoder Network
y = layers.Input(shape=(n_classes,))
masked_by_y = Mask()([capsule, y])  # True label is used to mask the output of capsule layer. 
masked = Mask()(capsule)  # Mask using the capsule with maximal length.

# Shared Decoder model in training and prediction
decoder = models.Sequential(name='decoder')
decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_classes))
decoder.add(layers.Dense(1024, activation='relu'))
decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

# Models for training and evaluation (prediction)
train_model = models.Model([inputs, y], [out_caps, decoder(masked_by_y)])
eval_model = models.Model(inputs, [out_caps, decoder(masked)])

# Compile the model
train_model.compile(optimizer='adam', loss=[margin_loss, 'mse'], loss_weights=[1., 0.392], metrics={'out_caps': 'accuracy'})
# Convert labels to one-hot encoding
y_train = to_categorical(train_dataset.get_labels(), n_classes)
y_val = to_categorical(val_dataset.get_labels(), n_classes)

# Train the model
train_model.fit(
    [train_images, y_train], [y_train, train_images],
    batch_size=32,
    epochs=50,
    validation_data=([val_images, y_val], [y_val, val_images])
)
