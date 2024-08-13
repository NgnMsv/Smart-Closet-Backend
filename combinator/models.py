from django.db import models


class Combination(models.Model):
    shirt = models.ForeignKey(to='closet.Wearable', related_name='shirt_set', on_delete=models.CASCADE)
    pants = models.ForeignKey(to='closet.Wearable', related_name='pants_set', on_delete=models.CASCADE)
    footwear = models.ForeignKey(to='closet.Wearable', related_name='footwear_set', on_delete=models.CASCADE)
    label = models.BooleanField(null=True, blank=True)
    
    class Meta:
        unique_together = ('shirt', 'pants', 'footwear')

    def __str__(self):
        return f'{self.shirt}, {self.pants}, {self.footwear}'