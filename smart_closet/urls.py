
from django.contrib import admin
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from combinator.viewset import CombinationViewSet, AccurateCombinationView
from closet.viewset import ClosetViewSet, WearableViewSet


router = DefaultRouter()
router.register(r'combinations', CombinationViewSet, basename='combination')
router.register(r'closets', ClosetViewSet, basename='closet')
router.register(r'wearables', WearableViewSet, basename='wearable')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/accurate-combination/<str:usage>/', AccurateCombinationView.as_view(), name='combination-detail'),
    path('api/', include(router.urls)),
    path('auth/', include('djoser.urls')),
    path('auth/', include('djoser.urls.jwt')),

]
