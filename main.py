import customtkinter as ctk
from customtkinter import CTkImage 
from tkinter import filedialog
from PIL import Image

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("dark-blue")

img_path = None

def upload_file():
    global img_path
    result_label.configure(text="", text_color="red")
    
    img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg")])
    if img_path:
        img = Image.open(img_path)
        img = img.resize((300, 200))
        ctk_img = CTkImage(light_image=img, dark_image=img, size=(300, 200)) 
        img_label.configure(image=ctk_img)
        img_label.image = ctk_img
        img_label.configure(text="")

def search_for_car_model():
    global img_path
    if not img_path:
        result_label.configure(text="Please upload a photo.", text_color="red")
        return

    result_label.configure(text="", text_color="red")
    result = "Tesla Model 3"
    
    result_label.configure(text=f"{result}", text_color="green")
    
def reset_app():
    global img_path
    img_path = None
    img_label.configure(image=None)
    img_label.image = None
    result_label.configure(text="No photo uploaded.")
    
    

app = ctk.CTk()
app.geometry("600x500")
app.title("Car Finder")

title_label = ctk.CTkLabel(app, text="Car Finder", font=ctk.CTkFont(size=20, weight="bold"))
title_label.pack(pady=20)

upload_button = ctk.CTkButton(app, text="upload photo", command=upload_file)
upload_button.pack(pady=20)

img_label = ctk.CTkLabel(app, text="No photo uploaded.", width=300, height=200, corner_radius=10)
img_label.pack(pady=20)

find_button = ctk.CTkButton(app, text="find model", command=search_for_car_model)
find_button.pack(pady=20)

# reset_button = ctk.CTkButton(app, text="reset", command=reset_app)
# reset_button.pack(pady=20)

result_label = ctk.CTkLabel(app, text="", font=ctk.CTkFont(size=16))
result_label.pack(pady=20)



app.mainloop()