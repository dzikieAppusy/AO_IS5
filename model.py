import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model = load_model('car_model_classifier.h5')

def search_for_car_model():
    if not img_path:
        result_label.configure(text="Please, upload photo.", text_color="red")
        return

    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    result = np.argmax(predictions)

    result_label.configure(text=f"{result}", text_color="green")
