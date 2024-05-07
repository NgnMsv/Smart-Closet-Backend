from django.db import models
from user.models import ClosetUser

class Closet(models.Model):
    name = models.CharField(max_length=255)
    user = models.ForeignKey(ClosetUser, on_delete=models.CASCADE)

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

