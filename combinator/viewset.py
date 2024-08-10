from rest_framework import viewsets
from combinator.models import Combination
from combinator.serializers import CombinationSerializer
from rest_framework.permissions import IsAuthenticated


class CombinationViewSet(viewsets.ModelViewSet):
    """
    This viewset automatically provides `list` and `retrieve` actions.
    """

    serializer_class = CombinationSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        # Filter combinations to only include those owned by the authenticated ClosetUser
        return Combination.objects.filter(user=self.request.user.closetuser)

    def perform_create(self, serializer):
        # Automatically associate the combination with the logged-in ClosetUser
        serializer.save(user=self.request.user.closetuser)