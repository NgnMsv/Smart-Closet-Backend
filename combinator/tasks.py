from .services import CombinatorServices, AIServices
from closet.models import ClosetUser
import logging
from celery import shared_task

@shared_task
def train_model():
    for user in ClosetUser.objects.all():
        try:
            ai_service = AIServices(user)
            ai_service.train_model()
            logging.info(f"Training for {user} is successfull!")
        except Exception as e:
            # logging.exception(e)
            ...