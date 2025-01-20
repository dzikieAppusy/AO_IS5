import os
import shutil

import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib
from imutils import paths
from pathlib import Path
from keras.src.metrics import TopKCategoricalAccuracy
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

IMAGE_DIR = './Dataset/imgs'
ATTR_PATH = './Dataset/companies.csv'
TRAIN_DIR = './Dataset/train'
VAL_DIR = './Dataset/val'

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

VAL_SPLIT = 0.25
BATCH_SIZE = 128
EPOCHS = 20  # 20
FINE_TUNE_EPOCHS = 35  # 25
matplotlib.use('TkAgg')



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


def generate_labels(image_paths, class_to_label):
    labels = [class_to_label[os.path.basename(os.path.dirname(path))] for path in image_paths]
    return labels


def get_class_labels(image_paths):
    class_names = sorted(set(os.path.basename(os.path.dirname(path)) for path in image_paths))

    class_to_label = {class_name: idx for idx, class_name in enumerate(class_names)}
    return class_to_label, class_names


def preprocess_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])

    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img


def create_dataset(paths, labels, batch_size, num_classes):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(lambda x, y: (preprocess_image(x), tf.one_hot(y, num_classes)))
    dataset = dataset.shuffle(buffer_size=len(paths)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


print("[INFO] Getting file paths and shuffling")
image_paths = list(sorted(paths.list_images(IMAGE_DIR)))

print("[INFO] Configuring training and testing data")
class_to_label, class_names = get_class_labels(image_paths)
num_classes = len(class_names)
labels = generate_labels(image_paths, class_to_label)
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, stratify=labels, test_size=VAL_SPLIT, shuffle=True, random_state=42
)

print("[INFO] Creating ImageFolder's for training and validation datasets")
create_image_folder_format(train_paths, TRAIN_DIR)
create_image_folder_format(val_paths, VAL_DIR)

train_dataset = create_dataset(train_paths, train_labels, BATCH_SIZE, num_classes)
val_dataset = create_dataset(val_paths, val_labels, BATCH_SIZE, num_classes)

print("Building model...")
base_model = tf.keras.applications.EfficientNetV2B0(
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    weights='imagenet',
)
base_model.trainable = False

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    # tf.keras.layers.RandomRotation(0.15),
    # tf.keras.layers.RandomZoom(0.15),
    # tf.keras.layers.RandomContrast(0.1),
    # tf.keras.layers.RandomBrightness(0.1),
    # tf.keras.layers.RandomCrop(height=IMG_HEIGHT, width=IMG_WIDTH),
])

model = tf.keras.Sequential([
    data_augmentation, # horizontal flip
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.015)), # 0.02
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.005)), # 0.01
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),

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

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy', TopKCategoricalAccuracy(k=3)]
)

print("Training model...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stopping, lr_scheduler],
    class_weight=class_weights_dict
)

print("Fine-tuning model...")
base_model.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy', TopKCategoricalAccuracy(k=3)]
)

fine_tune_history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[checkpoint, early_stopping, lr_scheduler],
    class_weight=class_weights_dict
)

print("Evaluating model...")
results = model.evaluate(val_dataset)
test_loss, test_accuracy, top_k_accuracy, *_ = results
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Top-K Accuracy: {top_k_accuracy}")

print("Saving model...")
model.save('car_model_classifier.keras')

plt.figure(figsize=(14, 6))

history_dict = history.history
epochs = np.arange(len(history_dict['loss']))
fine_tune_history_dict = fine_tune_history.history
fine_tune_epochs = np.arange(len(fine_tune_history_dict['loss']))

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
