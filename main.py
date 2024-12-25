import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk

# Initialize the customtkinter theme
ctk.set_appearance_mode("System")  # Options: "System", "Light", "Dark"
ctk.set_default_color_theme("blue")  # Options: "blue", "green", "dark-blue"

# Function to handle file upload
def upload_file():
    global img_label, uploaded_img_path
    uploaded_img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg")])
    if uploaded_img_path:
        # Load and display the image
        img = Image.open(uploaded_img_path)
        img = img.resize((300, 200))
        img = ImageTk.PhotoImage(img)
        img_label.configure(image=img)
        img_label.image = img  # Keep reference to avoid garbage collection
        prediction_label.configure(text="")  # Clear prediction

# Function to handle prediction
def predict_car_model():
    global uploaded_img_path
    if not uploaded_img_path:
        prediction_label.configure(text="Please upload an image first!", text_color="red")
        return

    # Simulated prediction for now
    predicted_model = "Simulated Model: Tesla Model 3"
    
    # Update the result
    prediction_label.configure(text=f"Predicted Car Model: {predicted_model}", text_color="green")

# Create the main application window
app = ctk.CTk()
app.geometry("600x500")
app.title("Car Model Finder")

# Add title
title_label = ctk.CTkLabel(app, text="Car Model Finder", font=ctk.CTkFont(size=20, weight="bold"))
title_label.pack(pady=20)

# Upload button
upload_button = ctk.CTkButton(app, text="Upload Image", command=upload_file)
upload_button.pack(pady=20)

# Image display area
img_label = ctk.CTkLabel(app, text="Your image will appear here", width=300, height=200, corner_radius=10)
img_label.pack(pady=20)

# Predict button
predict_button = ctk.CTkButton(app, text="Predict", command=predict_car_model)
predict_button.pack(pady=20)

# Prediction result label
prediction_label = ctk.CTkLabel(app, text="", font=ctk.CTkFont(size=16))
prediction_label.pack(pady=20)

# Run the application
app.mainloop()