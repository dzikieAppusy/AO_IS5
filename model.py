from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model
model = load_model('car_model_classifier.h5')

def predict_car_model():
    if not uploaded_img_path:
        prediction_label.configure(text="Please upload an image first!", text_color="red")
        return

    # Preprocess the uploaded image
    img = load_img(uploaded_img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict with the model
    predictions = model.predict(img_array)
    predicted_model = np.argmax(predictions)  # Replace with your label mapping logic

    # Update the result
    prediction_label.configure(text=f"Predicted Car Model: {predicted_model}", text_color="green")
