from rest_framework import serializers
from combinator.models import Combination


class CombinationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Combination
        fields = ['id', 'shirt', 'pants', 'footwear', 'label']
