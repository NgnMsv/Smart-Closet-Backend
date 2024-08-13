
from django.contrib import admin
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from combinator.viewset import CombinationViewSet
from closet.viewset import ClosetViewSet

router = DefaultRouter()
router.register(r'combinations', CombinationViewSet, basename='combination')
router.register(r'closets', ClosetViewSet, basename='closet')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include(router.urls)),
    path('auth/', include('djoser.urls')),
    path('auth/', include('djoser.urls.jwt')),

]
