import os
import shutil

import tensorflow as tf
import numpy as np
import matplotlib
from imutils import paths
from pathlib import Path
from keras.src.metrics import TopKCategoricalAccuracy
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

IMAGE_DIR = './Dataset/imgs'
ATTR_PATH = './Dataset/companies.csv'
TRAIN_DIR = './Dataset/train'
VAL_DIR = './Dataset/val'

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

VAL_SPLIT = 0.15
BATCH_SIZE = 64
EPOCHS = 10  # 30
FINE_TUNE_EPOCHS = 5  # 20


def create_image_folder_format(image_paths: str,
                               folder: str):
    data_path = Path(folder)
    if not data_path.is_dir():
        data_path.mkdir(parents=True, exist_ok=True)

    for path in image_paths:
        full_path = Path(path)
        image_name = full_path.name
        label = full_path.parent.name
        label_folder = data_path / label

        if not label_folder.is_dir():
            label_folder.mkdir(parents=True, exist_ok=True)

        destination = label_folder / image_name
        shutil.copy(path, destination)


# load all the image paths and split them into train & validation sets
print("[INFO] Getting file paths and shuffling")
image_paths = list(sorted(paths.list_images(IMAGE_DIR)))

print("[INFO] Configuring training and testing data")
class_names = [os.path.basename(os.path.dirname(x)) for x in image_paths]
train_paths, val_paths = train_test_split(image_paths, stratify=class_names, test_size=VAL_SPLIT, shuffle=True,
                                          random_state=42)

# copy the training and validation images to directories
print("[INFO] Creating ImageFolder's for training and validation datasets")
create_image_folder_format(train_paths, TRAIN_DIR)
create_image_folder_format(val_paths, VAL_DIR)

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True,
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    VAL_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False,
)

class_names = train_data.class_names
num_classes = len(class_names)

print("Building model...")
base_model = tf.keras.applications.EfficientNetV2B0(include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
base_model.trainable = False

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomContrast(0.1),
    tf.keras.layers.RandomBrightness(0.1),
])

model = tf.keras.Sequential([
    data_augmentation,
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.keras',
    save_best_only=True
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy', TopKCategoricalAccuracy(k=5)]
)

print("Training model...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stopping]
)

print("Fine-tuning model...")
base_model.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy', TopKCategoricalAccuracy(k=5)]
)

fine_tune_history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[checkpoint, early_stopping]
)

print("Evaluating model...")
results = model.evaluate(val_data)
test_loss, test_accuracy, top_k_accuracy, *_ = results
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Top-K Accuracy: {top_k_accuracy}")

print("Saving model...")
model.save('car_model_classifier.keras')

matplotlib.use('TkAgg')
plt.figure(figsize=(14, 6))

history_dict = history.history
fine_tune_history_dict = fine_tune_history.history
epochs = np.arange(EPOCHS)
fine_tune_epochs = np.arange(FINE_TUNE_EPOCHS)

plt.subplot(2, 2, 1)
plt.plot(epochs, history_dict['accuracy'], 'b-', label='Training Accuracy')
plt.plot(epochs, history_dict['val_accuracy'], 'r-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(epochs, history_dict['loss'], 'b-', label='Training Loss')
plt.plot(epochs, history_dict['val_loss'], 'r-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(fine_tune_epochs, fine_tune_history_dict['accuracy'], 'g-', label='Fine Tuning Training Accuracy')
plt.plot(fine_tune_epochs, fine_tune_history_dict['val_accuracy'], 'k-', label='Fine Tuning Validation Accuracy')
plt.title('Fine Tuning Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(fine_tune_epochs, fine_tune_history_dict['loss'], 'g-', label='Fine Tuning Training Loss')
plt.plot(fine_tune_epochs, fine_tune_history_dict['val_loss'], 'k-', label='Fine Tuning Validation Loss')
plt.title('Fine Tuning Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
