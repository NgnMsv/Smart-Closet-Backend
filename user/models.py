from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class ClosetUser(models.Model):
    first_name = models.CharField(max_length=255)
    last_name = models.CharField(max_length=255)
    email = models.CharField(max_length=255)
    phone_number = models.CharField(max_length=255)
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='closet_user')  

    def __str__(self) -> str:
        return f"{self.first_name} {self.last_name}"