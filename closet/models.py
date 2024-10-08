from django.db import models
from user.models import ClosetUser

class Closet(models.Model):
    name = models.CharField(max_length=255)
    user = models.ForeignKey(ClosetUser, on_delete=models.CASCADE)

    def __str__(self) -> str:
        return f"{self.user.first_name}, {self.name}"


class WearableManager(models.Manager):
    def shirts(self):
        """Return all Wearables of type 'shirt'."""
        return self.filter(type=Wearable.TypeChoices.shirt)

    def pants(self):
        """Return all Wearables of type 'pants'."""
        return self.filter(type=Wearable.TypeChoices.pants)

    def footwear(self):
        """Return all Wearables of type 'footwear'."""
        return self.filter(type=Wearable.TypeChoices.footwear)

# Create your models here.
class Wearable(models.Model):
    class TypeChoices(models.TextChoices):
        shirt = 's', 'SHIRT'
        pants = 'p', 'PANTS'
        footwear = 'f', 'FOOTWEAR'

    class UsageChoices(models.TextChoices):
        formal = 'f', 'FORMAL'
        casual = 'c', 'CASUAL'
        sport = 's', 'SPORT'
        general = 'g', 'GENERAL'

    closet = models.ForeignKey(Closet, on_delete=models.CASCADE)
    color = models.CharField(max_length=255, blank=True, null=True)
    type = models.CharField(max_length=8, choices=TypeChoices.choices)
    usage_1 = models.CharField(max_length=8, choices=UsageChoices.choices)
    usage_2 = models.CharField(max_length=8, choices=UsageChoices.choices)
    # image = models.ImageField(null=True, blank=True)
    image_url = models.URLField(max_length=500, null=True, blank=True)  # Store the Cloudinary URL here
    accessible = models.BooleanField(default=True)

    objects = WearableManager()

    def __str__(self) -> str:
        return f"{self.closet.name} <- {dict(self.TypeChoices.choices)[self.type]}:{self.color}:{dict(self.UsageChoices.choices)[self.usage_1]}:{dict(self.UsageChoices.choices)[self.usage_2]}"
