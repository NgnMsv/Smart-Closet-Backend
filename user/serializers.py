from djoser.serializers import UserCreateSerializer, UserSerializer
from .models import ClosetUser

class ClosetUserCreateSerializer(UserCreateSerializer):
    class Meta(UserCreateSerializer.Meta):
        model = ClosetUser
        fields = ('id', 'first_name', 'last_name', 'phone_number', 'email', 'password')
    
class ClosetUserSerializer(UserSerializer):
    class Meta(UserSerializer.Meta):
        model = ClosetUser
        fields = ('id', 'first_name', 'last_name', 'phone_number', 'email')
