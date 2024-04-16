from aiogram.filters.callback_data import CallbackData

class MacInfo(CallbackData, prefix='mac'):
    call_type: str
    call_but: str
    num: int
    
