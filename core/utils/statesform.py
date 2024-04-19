from aiogram.fsm.state import StatesGroup, State

class StepsForm(StatesGroup):
    GET_NAME = State()
    GET_LAST_NAME = State()
    GET_AGE = State()
    
class LossSteps(StatesGroup):
    GET_ANIMAL = State()
    GET_LOCATION_CONTACT = State()
    GET_DESCRIPTION = State()
    GET_PHOTO = State()
    
class FindSteps(StatesGroup):
    GET_ANIMAL = State()
    GET_LOCATION_CONTACT = State()
    GET_PHOTO = State()



