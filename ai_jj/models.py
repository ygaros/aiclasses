# Importuje klasę models z modułu django.db
from django.db import models
# Importuje default_storage z modułu django.core.files.storage
from django.core.files.storage import default_storage
# Importuje moduł image z tensorflow.keras.preprocessing i przypisuje mu alias tf_image
from tensorflow.keras.preprocessing import image as tf_image
# Importuje klasy i funkcje z inception_v3
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
# Importuje moduł numpy i przypisuje mu alias np
from django.core.files.base import ContentFile  # Importuje klasę ContentFile z django.core.files.base
import numpy as np
import openai
import os

openai.api_key = os.environ.get('OPENAI_API_KEY')
# Create your models here.
class Article(models.Model):
    title = models.CharField(max_length=255, blank=True)
    content = models.TextField(blank=True)
    photo = models.ImageField(upload_to="mediaphoto", blank=True, null=True)
    description = models.TextField(blank=True)
    def __str__(self):
        return self.title

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)

        if self.photo:
            try:  # Rozpoczyna blok try
                file_path = self.photo.path
                if default_storage.exists(file_path):
                    # Ładuje obraz o zadanych wymiarach
                    pil_image = tf_image.load_img(file_path, target_size=(299, 299))
                    # Konwertuje obraz na tablicę numpy
                    img_array = tf_image.img_to_array(pil_image)
                    # Rozszerza tablicę o nowy wymiar
                    img_array = np.expand_dims(img_array, axis=0)
                    # Przetwarza obraz zgodnie z wymaganiami modelu
                    img_array = preprocess_input(img_array)

                    # Tworzy model InceptionV3
                    model = InceptionV3(weights='imagenet')
                    # Dokonuje predykcji na obrazie
                    predictions = model.predict(img_array)
                    # Dekoduje predykcje
                    decoded_predictions = decode_predictions(predictions, top=1)[0]
                    # Wybiera najbardziej prawdopodobną etykietę
                    best_guess = decoded_predictions[0][1]
                    # Ustawia tytuł na najbardziej prawdopodobną etykietę
                    self.title = best_guess
                    self.description = self.generate_description()
                    # Tworzy łańcuch znaków zawierający etykiety i prawdopodobieństwa predykcji
                    self.content = ', '.join([f"{pred[1]}: {pred[2] * 100:.2f}%" for pred in decoded_predictions])
                    super().save(*args, **kwargs)

            except Exception as e:  # Obsługuje wyjątek
                # Nic nie robi w przypadku wystąpienia wyjątku
                print(e)
                pass

    def generate_description(self):
        try:
            # Wywołanie GPT-3 z użyciem klucza API, przekazując tytuł jako część monitu
            response = openai.Completion.create(
                # Wywołuje metodę create na obiekcie Completion z pakietu openai, aby uzyskać odpowiedź od modelu GPT-3
                engine="gpt-3.5-turbo-instruct",  # Określa model GPT-3 do wykorzystania
                prompt=f"Generate a descriptive text based on the following title: {self.title}\n",
                # Definiuje monit dla modelu GPT-3, prosząc o wygenerowanie opisu na podstawie tytułu
                temperature=0.7,  # Ustawia temperaturę dla procesu generowania, wpływając na kreatywność odpowiedzi
                max_tokens=100  # Określa maksymalną liczbę tokenów, które model może wygenerować jako odpowiedź
            )
            # Zwróć tekst wygenerowany przez GPT-3 jako opis
            return response.choices[
                0].text.strip()  # Zwraca tekst wygenerowany przez model, usuwając białe znaki z początku i końca
        except Exception as e:
            # W przypadku błędu zwróć komunikat błędu lub domyślny opis
            print(f"Error generating description: {str(e)}")  # Wypisuje komunikat o błędzie do konsoli
            return "{}".format(e)  # Zwraca komunikat błędu jako opis
