from django.db import models

class Book(models.Model):
    title = models.CharField(max_length=200)
    author = models.CharField(max_length=200, blank=True)
    date_read = models.DateField()
    notes = models.TextField(blank=True)

    def __str__(self):
        return self.title
