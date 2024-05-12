from django.db import models
from django.core.files.storage import default_storage
from tensorflow.keras.preprocessing import image as tf_image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from django.core.files.base import ContentFile
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
            try:
                file_path = self.photo.path
                if default_storage.exists(file_path):
                    pil_image = tf_image.load_img(file_path, target_size=(299, 299))
                    img_array = tf_image.img_to_array(pil_image)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_input(img_array)
                    model = InceptionV3(weights='imagenet')
                    predictions = model.predict(img_array)
                    decoded_predictions = decode_predictions(predictions, top=1)[0]
                    best_guess = decoded_predictions[0][1]
                    self.title = best_guess
                    self.description = self.generate_description()
                    self.content = ', '.join([f"{pred[1]}: {pred[2] * 100:.2f}%" for pred in decoded_predictions])
                    super().save(*args, **kwargs)

            except Exception as e:
                print(e)
                pass

    def generate_description(self):
        try:
            response = openai.Completion.create(
                engine="gpt-3.5-turbo-instruct",
                prompt=f"Generate a descriptive text based on the following title: {self.title}\n",
                temperature=0.7,
                max_tokens=100
            )
            return response.choices[
                0].text.strip()
        except Exception as e:
            print(f"Error generating description: {str(e)}")
            return "{}".format(e)
