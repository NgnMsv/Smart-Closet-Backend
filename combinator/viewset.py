from rest_framework import viewsets, status
from rest_framework.views import APIView
from combinator.models import Combination
from combinator.serializers import CombinationSerializer, CombinationCreateSerializer
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from combinator.services import CombinatorServices, AIServices
from django.shortcuts import get_object_or_404


class CombinationViewSet(viewsets.ModelViewSet):
    """
    This viewset automatically provides `list` and `retrieve` actions..
    """

    serializer_class = CombinationSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        # Filter combinations to only include those owned by the authenticated ClosetUser
        return Combination.objects.filter(shirt__closet__user=self.request.user)

    def perform_create(self, serializer):
        # Automatically associate the combination with the logged-in ClosetUser
        serializer.save(user=self.request.user)

    def create(self, request, *args, **kwargs):
        # Use the custom serializer for the POST request
        serializer = CombinationCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        combinator_service = CombinatorServices(request.user)
        try:
            combination = combinator_service.generate_random_set()
        except IndexError as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        result_serializer = self.get_serializer(combination)
        # Perform your custom logic here
        # Since no input is expected, you can proceed with any internal logic you need
        # For example, creating a combination or triggering a process

        # Example: Returning a simple response
        return Response(result_serializer.data, status=status.HTTP_201_CREATED)


class AccurateCombinationView(APIView):
    """
    API View that takes a combination ID, performs a calculation (sum of image URL lengths), and returns a number.
    """

    def post(self, request, *args, **kwargs):

        usage = self.kwargs.get('usage', None)
        if not usage:
            return Response({'error': 'The usage is not specified!'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Get the combination by its primary key (ID)
        combinator_service = CombinatorServices(request.user)
        ai_service = AIServices(request.user)

        combination, _ = combinator_service.generate_random_set_usage(usage=usage)
        accuracy = ai_service.predict_item(combination)
        while accuracy < 0.5:
            combination, _ = combinator_service.generate_random_set_usage(usage=usage)
            accuracy = ai_service.predict_item(combination)
            print(accuracy)
        
        # Return the result as a JSON response
        serializer = CombinationSerializer(combination)

        # Add the calculated result to the response data
        response_data = serializer.data
        response_data['accuracy'] = accuracy  # Add the calculated result to the response
        
        # Return the serialized data along with the calculated result
        return Response(response_data, status=status.HTTP_200_OK)