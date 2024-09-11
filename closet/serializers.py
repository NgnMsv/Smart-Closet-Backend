from rest_framework import serializers
from closet.models import Closet, Wearable


class ClosetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Closet
        fields = ['id', 'name']


class WearableSerializer(serializers.ModelSerializer):
    class Meta:
        model = Wearable
        fields = ['id', 'closet', 'color', 'type', 'usage_1', 'usage_2', 'image_url', 'accessible']

