# Klasyfikacja obiektów - wykrywanie modeli samochodów na postawie zdjęć .jpg

### 1. Założenia projektu
Projekt polegał na zrobieniu aplikacji desktopowej, która na podstawie załadowanego przez użytkownika zdjęcia samochodu w formacie .jpg dopasuje go do jednej z określonych wcześniej, w zestawie użytym do wytrenowania danego modelu, marek samochodów.

Aplikacja została w całości napisana w języku programowania Python. Można wyróżnić w niej 3 główne części: 
* **GUI** wykorzystujące moduły customtkinter, tkinter
* **model I PyTorch** stworzona przy użyciu modułów torch oraz torchvision
* **model II Keras** z wykorzystaniem API tensorflow.keras.



### 2. Podział pracy
* Anna Nowak - GUI, dokumentacja, testy manualne
* Emil Gruszecki - trenowanie modelu II, dokumentacja, testy manualne
* Sebastian Zarębski - trenowanie modelu I, dokumentacja, testy manualne


### 3. [Model I](https://github.com/dzikieAppusy/AO_IS5/tree/main/model-I)
Model I udało się wytrenować do dokładności treningowej około 93%, walidacyjnej oraz testowej w granicach 88-90%. Ten znajdujący się w projekcie ma właśnie taką dokładność. Model sprawdza dobrze w przypadku samochodów o bardzo charakterystycznym wyglądzie, takich jak przykładowo Nissan Leaf Hatchback 2012,

<p align="center">
  <img src="" />
  Zdjęcie 1. Poprawne rozpoznanie zdjęcia modelu Nissan Leaf Hatchback 2012, źródło obrazu Grafika Google.
</p>

jak również modeli samochodów o wyglądzie bardziej zbliżonym, np. model potrafi rozróżnić ze stosunkowo wysoką poprawnością Chevrolet Corvette Convertible 2012 oraz Ferrari California Convertible 2012. 

<p align="center">
  <img src="" />
  Zdjęcie 2. Poprawne rozpoznanie zdjęć modeli Chevrolet Corvette Convertible 2012 oraz Ferrari California Convertible 2012, źródło obrazu imgs_zip/cars_train.
</p>
<p align="center">
  <img src="" />
  Zdjęcie 3. Błędne rozpoznanie zdjęć modeli Chevrolet Corvette Convertible 2012 oraz Ferrari California Convertible 2012, źródło obrazu imgs_zip/cars_train.
</p>

#### Marki samochodów, które są rozpoznawane przez aplikację:
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

Program działa poprawnie dla znacznej większości danych znajdujących się w folderze imgs_zip/cars_train, oraz dla niektórych zdjęć wypisanych powyżej modeli pochodzących z wyszukiwarki Grafiki Google. Nie uda się natomiast uzyskać pozytywnego rezultatu jeżeli wykorzystamy zdjęcie modelu auta, którego rozpoznawania model PyTorch nie został nauczony. W przypadku użycia jakiegokolwiek zdjęcia niezwiązanego z modelami aut obiekt przedstawiony na zdjęciu również nie zostanie poprawnie rozpoznany.



### 4. [Model II](https://github.com/dzikieAppusy/AO_IS5/tree/main/model-II)



### 5. Wymagania niezbędne do uruchomienia aplikacji



### 6. Użytkowanie aplikacji
Aplikację uruchamiamy komendą python main.py znajdując się w głównym folderze projektu. Wyświetla się wtedy następujące okno:
<p align="center">
  <img src="https://github.com/user-attachments/assets/d628fbdf-e471-42cb-8fa9-713ce0576d6c" />
</p>
* Upload photo - przycisk odpowiadający za wybranie zdjęcia, wyskakuje okno, w którym wybieramy zdjęcie samochodu, które chcemy, aby model rozpoznał.
* Find model - przycisk, który wywołuje rozpoznawanie modelu przez aplikację i wypisuje na ekranie nazwę modelu samochodu.
* Reset - przycisk resetujący aplikację do stanu wejściowego.



### 7. Przykłady poprawnego działania aplikacji
* model I - z użyciem losowego zdjęcia samochodu pochodzącego z Grafiki Google, samochód należy do jednej z marek, do których rozpoznawania został wytrenowany model (Infiniti G Coupe IPL 2012)
<p align="center">
  <img src="https://github.com/user-attachments/assets/77228328-845f-4951-8a10-cb3309ba7015" />
</p>

*

### Źródła
* obrazy znajdujące się w folderze imgs_zip/cars_train należą do 
* 



