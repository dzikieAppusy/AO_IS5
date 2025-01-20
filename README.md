# AO_IS5_project

## Klasyfikacja obiektów
### Projekt polegał na zrobieniu aplikacji która na podstawie zdjęcia samochodu dopasuje go do jednej z określonych wcześniej marek samochodów.
Aplikacja składa się z dwóch części, GUI - stworzone przy użyciu Pythona oraz modułów customtkinter, tkinter oraz części uczącej model, stworzonej przy użyciu modułów torch oraz torchvision.
\
\
Model udało się wytrenować do dokładności treningowej około 93%, walidacyjnej oraz testowej w granicach 88-90%. Ten znajdujący się w projekcie ma właśnie taką dokładność
\
\
Podział na role:
- Anna Nowak - GUI, dokumentacja
* Emil Gruszecki - Trenowanie modelu, dokumentacja
+ Sebastian Zarębski - Trenowanie modelu, dokumentacja  
### Marki samochodów które są rozpoznawane przez aplikację:
```
Aston Martin Virage Coupe 2012
Audi R8 Coupe 2012
Audi TTS Coupe 2012
Bentley Mulsanne Sedan 2011
BMW 6 Series Convertible 2007
Cadillac CTS-V Sedan 2012
Chevrolet Corvette Convertible 2012
Chevrolet Malibu Sedan 2007
Daewoo Nubira Wagon 2002
Dodge Ram Pickup 3500 Crew Cab 2010
Ferrari California Convertible 2012
FIAT 500 Convertible 2012
Fisker Karma Sedan 2012
Ford Focus Sedan 2007
Geo Metro Convertible 1993
GMC Savana Van 2012
Honda Odyssey Minivan 2012
Infiniti G Coupe IPL 2012
Mercedes-Benz C-Class Sedan 2012
Nissan Leaf Hatchback 2012
```
\
\
Program na pewno działa poprawnie dla danych znajdujących się w folderze imgs_zip/cars_train, oraz dla niektórych zdjęć danych modeli wziętych z grafiki google. Nie działa natomiast jeżeli wrzucimy do niego zdjęcie modelu auta, do którego rozpoznawania nie został nauczony lub jakiekolwiek zdjęcie niezwiązane z modelami aut.
## Aplikacja
Aplikację uruchamiamy komendą python main.py znajdując się w głównym folderze projektu. Wyświetla się wtedy następujące okno:
![image](https://github.com/user-attachments/assets/d628fbdf-e471-42cb-8fa9-713ce0576d6c)
- Upload photo - przycisk odpowiadający za wybranie zdjęcia, wyskakuje okno, w którym wybieramy zdjęcie samochodu, które chcemy, aby model rozpoznał.
* Find model - przycisk, który wywołuje rozpoznawanie modelu przez aplikację i wypisuje na ekranie nazwę modelu samochodu.
+ Reset - przycisk resetujący aplikację do stanu wejściowego.
### Przykład działania na losowym zdjęciu z grafiki google samochodu należącego do marek które rozpoznaje (Infiniti G Coupe IPL 2012):
![image](https://github.com/user-attachments/assets/77228328-845f-4951-8a10-cb3309ba7015)



