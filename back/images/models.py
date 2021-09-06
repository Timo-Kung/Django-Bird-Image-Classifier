from django.db import models
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np
import cv2
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, decode_predictions, preprocess_input

# Create your models here.
class Image(models.Model):
    picture = models.ImageField()
    classified = models.CharField(max_length=200, blank=True)
    uploaded = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return "Image classfied at {}".format(self.uploaded.strftime('%Y-%m-%d %H:%M'))

    def save(self, *args, **kwargs):
        try:
           
           img = load_img(str(self.picture), target_size=(150,150))
           img_array = img_to_array(img)
           to_pred = np.expand_dims(img_array, axis=0)
           prep = preprocess_input(to_pred)
           model = load_model('./Bird-13-types_2.h5')
           prediction = model.predict(prep)
           decoded = decode_predictions(prediction)[0][0][1]
           self.classified = str(decoded)
           print('success')

        except Exception as e:
            print('classification failed', e)
        super().save(*args, **kwargs)
