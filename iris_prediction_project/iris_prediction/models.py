from django.db import models

class IrisData(models.Model):
    sepal_length = models.FloatField()
    sepal_width = models.FloatField()
    petal_length = models.FloatField()
    petal_width = models.FloatField()

class Prediction(models.Model):
    model_used = models.CharField(max_length=100)
    predicted_class = models.CharField(max_length=100)
    
