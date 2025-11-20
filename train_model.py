import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split

IMAGE_SIZE = 100
DATASET_DIR = "Dataset"

CLASSES = ["Fist", "Palm", "Swing"]

def load_images(folder, label):
    data = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img_to_array(img)
        data.append((img, label))
    return data

def load_dataset():
    dataset = []

    for idx, gesture in enumerate(CLASSES):
        train_folder = os.path.join(DATASET_DIR, f"{gesture}Images")
        test_folder  = os.path.join(DATASET_DIR, f"{gesture}Test")

        dataset += load_images(train_folder, idx)
        dataset += load_images(test_folder, idx)

    images, labels = zip(*dataset)
    images = np.array(images, dtype="float32") / 255.0
    labels = to_categorical(labels, num_classes=len(CLASSES))
    return images, labels

print("Loading dataset...")
X, y = load_dataset()
print(f"Dataset loaded: {len(X)} samples")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(CLASSES), activation='softmax')
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

print("Training model...")
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

print("Saving model...")
model.save("gesture_model.h5")
print("Model saved as gesture_model.h5")
