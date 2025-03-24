import os
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset Parameters
img_width, img_height = 224, 224
batch_size = 32
epochs = 10
dataset_path = "DataSet"
model_file = "model.keras"
feature_model_file = "feature_model.keras"

# Ensure dataset path exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset path '{dataset_path}' does not exist.")

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Class Indices
class_indices = train_generator.class_indices
labels = {v: k for k, v in class_indices.items()}
np.save("class_labels.npy", labels)

# Load Pre-trained Model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))

# Freeze Base Model Layers
for layer in base_model.layers:
    layer.trainable = False

# Add Custom Layers for Classification
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dense(len(class_indices), activation='softmax')
])

# Compile the Classification Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the Model
model.fit(train_generator, epochs=epochs)

# Save Classification Model
model.save(model_file)
print(f"Classification model saved to {model_file}")

# Save Feature Extraction Model
feature_extractor = Sequential([
    base_model,
    GlobalAveragePooling2D()
])
feature_extractor.save(feature_model_file)
print(f"Feature extraction model saved to {feature_model_file}")