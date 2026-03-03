# train_cnn_model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from preprocess_landmarks import train_generator, val_generator

NUM_CLASSES = len(train_generator.class_indices)

# ====== Define CNN Model ======
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("🚀 Training CNN model...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,  # Increase to 30 for better accuracy
    verbose=1
)

# ====== Save Model ======
model.save("asl_cnn_model.h5")
print("✅ Model saved as asl_cnn_model.h5")
