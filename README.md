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
Model I udało się wytrenować do dokładności treningowej około 93%, walidacyjnej oraz testowej w granicach 88-90%. Model I, oparty na sieci neuronowej MobileNetV2, ma właśnie taką dokładność. Model przyjmuje obrazy RGB o stałym wymiarze 224x224 piksele, dlatego też przed jego zastosowaniem rozmiar testowanego obrazu musi zostać dostosowany do wymaganych wymiarów. Model sprawdza dobrze w przypadku samochodów o bardzo charakterystycznym wyglądzie, takich jak przykładowo Nissan Leaf Hatchback 2012,

<p align="center">
  <img src="./readme-zdj/zdj1.png" />
  <br />
  Zdjęcie 1. Poprawne rozpoznanie zdjęcia modelu Nissan Leaf Hatchback 2012, źródło obrazu Grafika Google.
</p>

jak również modeli samochodów o wyglądzie bardziej zbliżonym, np. model potrafi rozróżnić ze stosunkowo wysoką poprawnością Chevrolet Corvette Convertible 2012 oraz Ferrari California Convertible 2012. 

<p align="center">
  <img src="./readme-zdj/zdj2.png" />
  <br />
  Zdjęcie 2. Poprawne rozpoznanie zdjęć modeli Chevrolet Corvette Convertible 2012 oraz Ferrari California Convertible 2012, źródło obrazu model-I/imgs_zip/cars_train.
</p>
<p align="center">
  <img src="./readme-zdj/zdj3.png" />
  <br />
  Zdjęcie 3. Błędne rozpoznanie zdjęcia modelu Chevrolet Corvette Convertible 2012, źródło obrazu model-I/imgs_zip/cars_train.
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
W pliku [requirements.txt](https://github.com/dzikieAppusy/AO_IS5/blob/main/requirements.txt) znajdują się informacje dotyczące modułów wykorzystanych do wytrenowania modelu oraz zbudowania aplikacji desktopowej.



### 6. Użytkowanie aplikacji


<p align="center">
  <img src="./readme-zdj/zdj5.png" />
  <br />
  Zdjęcie 4. Okno aplikacji desktopowej <I>Car Finder</I>.
</p>

* **upload photo** - przycisk odpowiadający za załadowanie zdjęcia. Otwiera okno, w którym użytkownik może wybrać z eksploratora plików zdjęcie samochodu w formacie .jpg, które chce aby model rozpoznał
* **find - model I** - przycisk, który wywołuje akcję rozpoznawania modelu przy użuciu Modelu I i wypisuje na ekranie nazwę rozpoznanego modelu samochodu. W przypadku niepowodzenia operacji wyświetlona zostanie informacja o błędzie
* **find - model II** - przycisk, który wywołuje akcję rozpoznawania modelu przy użuciu Modelu II i wypisuje na ekranie nazwę rozpoznanego modelu samochodu. W przypadku niepowodzenia operacji wyświetlona zostanie informacja o błędzie
* **reset** - przycisk usuwający wczytany aktualnie obraz z pamięci aplikacji i resetujący aplikację do stanu początkowego.



### 7. Przykłady poprawnego działania aplikacji
* model I - z użyciem losowego zdjęcia samochodu pochodzącego z Grafiki Google, samochód należy do jednej z marek, do których rozpoznawania został wytrenowany model (Infiniti G Coupe IPL 2012)
<p align="center">
  <img src="./readme-zdj/zdj4.png" />
</p>

* model II - 



### 8. Źródła
* obrazy znajdujące się w folderze model-I/imgs_zip/cars_train należą do Stanford Cars Dataset, wybrano z niego 20 klas - https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset
* pliki z model-II/
* zawartość folderu zdj-testowe-grafika-google pochodzą z Grafiki Google



