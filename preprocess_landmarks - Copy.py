# preprocess_images.py
import os
import numpy as np
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ====== CONFIG ======
DATA_DIR = r"C:/Users/hanns/Downloads/ASL/asl_alphabet_test"  # <-- Change if needed
IMG_SIZE = (64, 64)  # Resize images to 64x64
BATCH_SIZE = 32

# ====== Data Augmentation & Preprocessing ======
datagen = ImageDataGenerator(
    rescale=1./255,           # Normalize pixel values to [0,1]
    rotation_range=10,        # Random rotation
    width_shift_range=0.1,    # Horizontal shift
    height_shift_range=0.1,   # Vertical shift
    zoom_range=0.1,           # Zoom
    validation_split=0.2      # 20% validation
)

# ====== Training Generator ======
train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# ====== Validation Generator ======
val_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# ====== Save class indices for inference ======
with open("class_indices.pkl", "wb") as f:
    pickle.dump(train_generator.class_indices, f)

print("✅ Preprocessing done. Class indices saved to class_indices.pkl")
