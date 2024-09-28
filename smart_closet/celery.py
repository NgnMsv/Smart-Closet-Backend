import os
from celery import Celery
from celery.schedules import crontab
from celery.schedules import schedule
from datetime import timedelta

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'smart_closet.settings')
app = Celery('smart_closet')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()


@app.task(bind=True, ignore_result=True)
def debug_task(self):
    print(f'Request: {self.request!r}')


app.conf.beat_schedule = {
    # 'run-my-scheduled-task-every-30-minutes': {
    #     'task': 'smart_closet.celery.debug_task',
    #     'schedule': crontab(minute='*/1'),
    # },
    'train-model': {
        'task': 'combinator.tasks.train_model',
        #  'schedule': crontab(second='*/5'),
        'schedule': schedule(run_every=timedelta(seconds=5)),

    },
}
