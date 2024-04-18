from aiogram.filters.callback_data import CallbackData

class MacInfo(CallbackData, prefix='mac'):
    call_type: str
    call_but: str
    num: int
    
class InlineInfo(CallbackData, prefix='inline'):
    type: str
    animal: str
    IsLocated: bool
    IsContact: bool

class MissingReport():
    cat_or_dog: int
    first_coord: float
    second_coord: float
    description: str
    path: str
    date: str
    id_tg: int
    phone_num: int

class StreetPets():
    cat_or_dog: int
    first_coord: float
    second_coord: float
    path: str
    date: str
    id_tg: int
    
