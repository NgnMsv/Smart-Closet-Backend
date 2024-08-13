from rest_framework import serializers
from closet.models import Closet


class ClosetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Closet
        fields = ['id', 'name']

