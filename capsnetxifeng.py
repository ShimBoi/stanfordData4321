import pandas as pd
import os
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np

# Load the Excel file
excel_path = '/root/stanfordData4321/stanfordData4321/dataRef/release_midas.xlsx'
df = pd.read_excel(excel_path)

# Filter out rows with NaN labels
df = df.dropna(subset=['clinical_impression_1'])

# Load image paths and labels
image_paths = []
labels = []

# List of root directories
root_dirs = ['/root/stanfordData4321/stanfordData4321/images1', 
             '/root/stanfordData4321/stanfordData4321/images2', 
             '/root/stanfordData4321/stanfordData4321/images3', 
             '/root/stanfordData4321/stanfordData4321/images4']
augmented_dir = '/root/stanfordData4321/stanfordData4321/augmented_images'

for _, row in df.iterrows():
    found = False
    for root_dir in root_dirs:
        image_path = os.path.join(root_dir, row['file_name'])
        if os.path.exists(image_path):
            image_paths.append(image_path)
            labels.append(row['clinical_impression_1'])
            found = True
            break
    if not found:
        # Check in the augmented directory
        image_path = os.path.join(augmented_dir, row['file_name'])
        if os.path.exists(image_path):
            image_paths.append(image_path)
            labels.append(row['clinical_impression_1'])

# Convert labels to categorical
labels = to_categorical(labels, num_classes=13)

# Load and preprocess images
def load_image(img_path, target_size):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(target_size)
    img = np.array(img).astype('float32') / 255.0
    return img

x_data = np.array([load_image(img, (width, height)) for img in image_paths])
y_data = np.array(labels)

# Custom data loading function
def load_custom_data():
    # Use x_data and y_data from the preprocessing step
    split_idx = int(0.8 * len(x_data))
    x_train, y_train = x_data[:split_idx], y_data[:split_idx]
    x_test, y_test = x_data[split_idx:], y_data[split_idx:]
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_custom_data()
print(len(x_train))
print(len(x_test))
model, eval_model, manipulate_model = CapsNet(input_shape=(height, width, 3),
                                              n_class=13,
                                              routings=args.routings)
import keras
from keras import callbacks
from capsulenet import CapsNet, margin_loss  # Assuming you have the CapsNet architecture in capsulenet.py

# Training parameters
epochs = 5
batch_size = 32

# Model, eval_model, and manipulate_model are obtained from CapsNet function as shown before
model.compile(optimizer=keras.optimizers.Adam(lr=1e-3),
              loss=[margin_loss, 'mse'],
              loss_weights=[1., 0.392])

# Define callbacks
log = callbacks.CSVLogger('log.csv')
tb = callbacks.TensorBoard(log_dir='./tensorboard-logs', batch_size=batch_size, histogram_freq=1)
checkpoint = callbacks.ModelCheckpoint('weights-{epoch:02d}.h5', save_best_only=True, save_weights_only=True, verbose=1)
lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: 0.001 * np.power(0.9, epoch))

# Train the model
model.fit([x_train, y_train], [y_train, x_train],
          batch_size=batch_size,
          epochs=epochs,
          validation_data=[[x_test, y_test], [y_test, x_test]],
          callbacks=[log, tb, checkpoint, lr_decay])
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
from keras.models import Model
from keras.preprocessing.image import img_to_array

def get_gradcam_image(model, img, label_index):
    # Define the model that outputs the activations of the last conv layer
    last_conv_layer = model.get_layer('conv2d')  # Replace 'conv2d' with the actual last conv layer name
    heatmap_model = Model([model.inputs], [last_conv_layer.output, model.output])
    
    # Get the gradient of the loss with respect to the output feature map
    with tf.GradientTape() as tape:
        conv_output, predictions = heatmap_model(np.array([img]))
        loss = predictions[:, label_index]
    
    grads = tape.gradient(loss, conv_output)[0]

    # Compute guided gradients
    guided_grads = K.sign(grads) * K.relu(grads)
    
    # Compute the heatmap
    conv_output = conv_output[0]
    guided_grads = guided_grads[0]
    weights = np.mean(guided_grads, axis=(0, 1))
    cam = np.dot(conv_output, weights)
    
    # Normalize the heatmap
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    
    return cam

# Generate a Grad-CAM heatmap
img = x_test[0]  # Replace with the actual image you want to visualize
label_index = np.argmax(y_test[0])  # Replace with the actual label index for this image

heatmap = get_gradcam_image(model, img, label_index)

# Display the heatmap
plt.imshow(img)
plt.imshow(heatmap, cmap='jet', alpha=0.5)
plt.show()

