from aiogram.fsm.state import StatesGroup, State
    
class LossSteps(StatesGroup):
    GET_ANIMAL = State()
    GET_LOCATION = State()
    GET_CONTACT = State()
    GET_DESCRIPTION = State()
    GET_PHOTO = State()
    
class FindSteps(StatesGroup):
    GET_ANIMAL = State()
    GET_LOCATION = State()
    GET_CONTACT = State()
    GET_DESCRIPTION = State()
    GET_PHOTO = State()



