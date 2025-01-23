import os
import tensorflow as tf
import numpy as np
import matplotlib
from keras.src.metrics import TopKCategoricalAccuracy
from tensorflow.keras import regularizers
from tensorflow.keras.losses import CategoricalCrossentropy
from keras_preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

IMAGE_DIR = 'CompCars/image'
LABEL_DIR = 'CompCars/label'
ATTR_PATH = 'CompCars/misc/attributes.txt'
TRAIN_SPLIT = 'CompCars/train_test_split/classification/train.txt'
TEST_SPLIT = 'CompCars/train_test_split/classification/test.txt'
IMG_SIZE = (384, 384)  # 384, 384
BATCH_SIZE = 16
EPOCHS = 30  # 30
FINE_TUNE_EPOCHS = 20  # 20


def load_split_list(file_path):
    """Load file list from split file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


def load_image(image_path, target_size, bbox=None):
    """Load and preprocess a single image. Optionally crop to bounding box."""
    img = load_img(image_path)
    img_array = img_to_array(img)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        img_array = img_array[y1:y2, x1:x2]

    img_array = tf.image.resize(img_array, target_size)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array


def load_attributes(attributes_path):
    """Load attributes into a dictionary for quick lookup."""
    attributes = {}
    with open(attributes_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            model_id = parts[0]
            attributes[model_id] = {
                'maximum_speed': parts[1],
                'displacement': parts[2],
                'door_number': parts[3],
                'seat_number': parts[4],
                'type': parts[5]
            }
    return attributes


def load_label(label_path, attributes):
    """Load label information from a label file and attributes."""
    with open(label_path, 'r') as f:
        lines = f.readlines()
    viewpoint = int(lines[0].strip())
    bbox = list(map(int, lines[2].strip().split()))

    normalized_path = os.path.normpath(label_path)
    parts = normalized_path.split(os.sep)

    make_id = parts[-4]
    model_id = parts[-3]
    released_year = parts[-2]

    attr = attributes.get(model_id, {})
    label = {
        'make_id': make_id,
        'model_id': model_id,
        'released_year': released_year,
        'viewpoint': viewpoint,
        'attributes': attr
    }
    return label, bbox


def extract_label_from_path(path, attributes):
    """Helper function to extract the label from the label file using load_label."""
    label_path = os.path.join(LABEL_DIR, path.replace('jpg', 'txt'))
    label, _ = load_label(label_path, attributes)
    label_key = f"{label['make_id']}_{label['model_id']}"
    return label_key


def preprocess_dataset(image_paths, labels, attributes):
    """Process the dataset and prepare it for training."""
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def preprocess(image_path, label_num):
        def load_label_and_bbox_py(label_num):
            label_path_str = os.path.join(LABEL_DIR, image_paths[label_num]).replace('jpg', 'txt')
            label, bbox = load_label(label_path_str, attributes)
            label_key = f"{label['make_id']}_{label['model_id']}"
            encoded_label = label_to_index[label_key]
            return encoded_label, bbox

        encoded_label, bbox = tf.py_function(
            func=load_label_and_bbox_py,
            inp=[label_num],
            Tout=[tf.int32, tf.int32]
        )
        encoded_label.set_shape([])
        bbox.set_shape([4])

        def load_image_py(image_path, bbox):
            image_path_str = os.path.join(IMAGE_DIR, image_path.numpy().decode('utf-8'))
            return load_image(image_path_str, IMG_SIZE, bbox)

        image = tf.py_function(
            func=load_image_py,
            inp=[image_path, bbox],
            Tout=tf.float32
        )
        image.set_shape([*IMG_SIZE, 3])

        encoded_label = tf.one_hot(encoded_label, len(label_to_index))

        return image, encoded_label

    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


attributes = load_attributes(ATTR_PATH)

print("Loading dataset...")
train_list = load_split_list(TRAIN_SPLIT)
test_list = load_split_list(TEST_SPLIT)

print("Encoding labels...")
unique_labels = sorted(set(extract_label_from_path(path, attributes) for path in (train_list + test_list)))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
train_labels_encoded = [label_to_index[extract_label_from_path(path, attributes)] for path in train_list]
test_labels_encoded = [label_to_index[extract_label_from_path(path, attributes)] for path in test_list]
num_classes = len(label_to_index)

print("Splitting validation set...")
train_list, val_list, train_labels_encoded, val_labels_encoded = train_test_split(
    train_list, train_labels_encoded, test_size=0.3, random_state=42
)

print("Creating datasets...")
train_dataset = preprocess_dataset(train_list, train_labels_encoded, attributes)
val_dataset = preprocess_dataset(val_list, val_labels_encoded, attributes)
test_dataset = preprocess_dataset(test_list, test_labels_encoded, attributes)

print("Building model...")
# base_model = tf.keras.applications.EfficientNetV2B0(include_top=False, input_shape=(*IMG_SIZE, 3))
base_model = tf.keras.applications.EfficientNetV2S(include_top=False, input_shape=(*IMG_SIZE, 3))
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
    tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.L2(0.02)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=1e-4,
    first_decay_steps=2000,
    t_mul=2.0,
    m_mul=1.0,
    alpha=0.1
)
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
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stopping]
)

print("Fine-tuning model...")
base_model.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy', TopKCategoricalAccuracy(k=5)]
)

fine_tune_history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[checkpoint, early_stopping]
)

print("Evaluating model...")
results = model.evaluate(test_dataset)
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
