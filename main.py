import customtkinter as ctk
from customtkinter import CTkImage 
from tkinter import filedialog, PhotoImage
from PIL import Image, ImageTk
import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

img_path = None
# model = load_model('model.h5')

def upload_file():
    global img_path
    result_label.configure(text="")
    
    img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg")])
    if img_path:
        img = Image.open(img_path)
        
        original_width, original_height = img.size
        new_height = 200
        new_width = int(original_width * (new_height / original_height))
        
        if new_width > 400:
            new_width = 400
            new_height = int(original_height * (new_width / original_width))
            
        img = img.resize((new_width, new_height))
        
        ctk_img = CTkImage(light_image=img, dark_image=img, size=(new_width, new_height)) 
        img_label.configure(image=ctk_img)
        img_label.image = ctk_img
        img_label.configure(text="")

def search_for_car_model():
    global img_path
    if not img_path:
        result_label.configure(text="Please upload a photo.", text_color="red")
        return

    result_label.configure(text="")
    
    # img = load_img(img_path, target_size=(224, 224))
    # img_array = img_to_array(img) / 255.0
    # img_array = np.expand_dims(img_array, axis=0)

    # predictions = model.predict(img_array)
    # result = np.argmax(predictions)
    result = "Tesla Model 3"
    
    result_label.configure(text=f"{result}", text_color="white")
    
def reset_app():
    global img_path
    img_path = None
    result_label.configure(text="")
    
    if hasattr(img_label, 'image') and img_label.image is not None:
        img_label.configure(image=None, text="No photo uploaded.")
        img_label.image = None



app = ctk.CTk()
app.geometry("450x650")
app.title("Car Finder")
app.resizable(False, False)

carF = Image.open("car.png")
carF = carF.resize((50, 28))
favicon = ImageTk.PhotoImage(carF) 
app.iconphoto(False, favicon)

canvas = ctk.CTkCanvas(app, height=100, bg="#0508a6", bd=0, highlightthickness=0)
canvas.pack(fill="x", padx=0, pady=0)

title_label = ctk.CTkLabel(app, text="Car Finder", font=ctk.CTkFont(size=28, weight="bold"), fg_color="#0508a6")
title_label.place(x=20, y=35) 

car = Image.open("car.png")
car = car.resize((100, 56))
car_icon = Image.new("RGB", (100, 56), "#0508a6")
car_icon.paste(car, (0, 0), car) 
car_icon_ctk = CTkImage(light_image=car_icon, dark_image=car_icon, size=(100, 56))
icon_label = ctk.CTkLabel(app, image=car_icon_ctk, text="")
icon_label.place(x=320, y=40) 

upload_button = ctk.CTkButton(app, text="upload photo", command=upload_file, width=200, height=40, corner_radius=10, font=ctk.CTkFont(size=18))
upload_button.pack(pady=20)

img_label = ctk.CTkLabel(app, text="No photo uploaded.", font=ctk.CTkFont(size=14), width=300, height=200)
img_label.pack(pady=20)

result_label = ctk.CTkLabel(app, text="", font=ctk.CTkFont(size=18))
result_label.pack(pady=20)

find_button = ctk.CTkButton(app, text="find model", command=search_for_car_model, width=200, height=40, corner_radius=10, font=ctk.CTkFont(size=18))
find_button.pack(pady=20)

reset_button = ctk.CTkButton(app, text="reset", command=reset_app, width=200, height=40, corner_radius=10, font=ctk.CTkFont(size=18))
reset_button.pack(pady=20)



app.mainloop()