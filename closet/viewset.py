from rest_framework import viewsets, status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from closet.serializers import ClosetSerializer, WearableSerializer
from closet.models import Closet, Wearable
from django_filters.rest_framework import DjangoFilterBackend



class ClosetViewSet(viewsets.ModelViewSet):
    """
    This viewset automatically provides `list` and `retrieve` actions.
    """

    serializer_class = ClosetSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        # Filter combinations to only include those owned by the authenticated ClosetUser
        return Closet.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        # Automatically associate the combination with the logged-in ClosetUser
        serializer.save(user=self.request.user)


class WearableViewSet(viewsets.ModelViewSet):

    serializer_class = WearableSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['closet']

    def get_queryset(self):
        # Filter combinations to only include those owned by the authenticated ClosetUser
        return Wearable.objects.filter(closet__user=self.request.user)

    # def perform_create(self, serializer):
    #     # Automatically associate the combination with the logged-in ClosetUser
    #     serializer.save(closet__user=self.request.user)

    