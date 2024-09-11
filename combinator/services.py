from closet.models import Closet, ClosetUser, Wearable
from random import choice
from combinator.models import Combination


class CombinatorServices:
    def __init__(self, user: ClosetUser):
        self._user = user

    def generate_random_set(self) -> Combination:
        shirt_ids = list(Wearable.objects.filter(closet__user=self._user, 
                                                 type=Wearable.TypeChoices.shirt,
                                                 accessible=True).\
                                                        values_list('id', flat=True))
        pants_ids = list(Wearable.objects.filter(closet__user=self._user, 
                                                 type=Wearable.TypeChoices.pants,
                                                 accessible=True).\
                                                        values_list('id', flat=True))
        footwear_ids = list(Wearable.objects.filter(closet__user=self._user, 
                                                    type=Wearable.TypeChoices.footwear,
                                                    accessible=True).\
                                                        values_list('id', flat=True))
        
        tries = 0
        random_shirt_id, random_pants_id, random_footwear_id = choice(shirt_ids), choice(pants_ids), choice(footwear_ids)
        while Combination.objects.filter(shirt_id=random_shirt_id, pants_id=random_pants_id, footwear_id=random_footwear_id).exists():
            random_shirt_id, random_pants_id, random_footwear_id = choice(shirt_ids), choice(pants_ids), choice(footwear_ids)
            tries += 1
            if tries > 20:
                raise IndexError("Cannot find enough wearables")
            
        combination = Combination.objects.create(shirt_id=random_shirt_id, 
                                                 pants_id=random_pants_id, 
                                                 footwear_id=random_footwear_id)
        return combination