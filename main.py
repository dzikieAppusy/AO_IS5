import customtkinter as ctk
from customtkinter import CTkImage 
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import mobilenet_v2

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")
CLASS_AMOUNT=20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = mobilenet_v2(pretrained=False)
num_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_features, CLASS_AMOUNT) 
model.load_state_dict(torch.load("model-I/final_model.pth", map_location=device))
model.eval()
model.to(device)

# Definicja transformacji
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
classes = ["Aston Martin Virage Coupe 2012", "Audi R8 Coupe 2012", "Audi TTS Coupe 2012", "BMW 6 Series Convertible 2007", "Bentley Mulsanne Sedan 2011", "Cadillac CTS-V Sedan 2012", "Chevrolet Corvette Convertible 2012", "Chevrolet Malibu Sedan 2007", "Daewoo Nubira Wagon 2002", "Dodge Ram Pickup 3500 Crew Cab 2010", "FIAT 500 Convertible 2012", "Ferrari California Convertible 2012", "Fisker Karma Sedan 2012","Ford Focus Sedan 2007", "GMC Savana Van 2012", "Geo Metro Convertible 1993", "Honda Odyssey Minivan 2012", "Infiniti G Coupe IPL 2012", "Mercedes-Benz C-Class Sedan 2012", "Nissan Leaf Hatchback 2012"]

img_path = None

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

def search_for_car_model_1():
    global img_path
    if not img_path:
        result_label.configure(text="Please upload a photo.", text_color="red")
        return

    result_label.configure(text="")
    try:
        # Przetwarzanie obrazu
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Przewidywanie
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            result = classes[predicted_idx.item()]

        result_label.configure(text=f"{result}", text_color="white")
    except Exception as e:
        result_label.configure(text="Error in prediction.", text_color="red")
        print(e)
        
def search_for_car_model_2():
    global img_path
    if not img_path:
        result_label.configure(text="Please upload a photo.", text_color="red")
        return

    result_label.configure(text="")
    try:
        # Przetwarzanie obrazu
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Przewidywanie
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            result = classes[predicted_idx.item()]

        result_label.configure(text=f"{result}", text_color="white")
    except Exception as e:
        result_label.configure(text="Error in prediction.", text_color="red")
        print(e)        
    
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

find_button_1 = ctk.CTkButton(app, text="find - model I", command=search_for_car_model_1, width=200, height=40, corner_radius=10, font=ctk.CTkFont(size=18))
find_button_1.pack(pady=20)

find_button_2 = ctk.CTkButton(app, text="find - model II", command=search_for_car_model_2, width=200, height=40, corner_radius=10, font=ctk.CTkFont(size=18))
find_button_2.pack(pady=20)

reset_button = ctk.CTkButton(app, text="reset", command=reset_app, width=200, height=40, corner_radius=10, font=ctk.CTkFont(size=18))
reset_button.pack(pady=20)



app.mainloop()