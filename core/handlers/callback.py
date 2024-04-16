from aiogram import Bot
from aiogram.types import CallbackQuery
from core.utils.callbackdata import MacInfo

# async def select_macbook(call: CallbackQuery, bot: Bot):
#     call_type, call_but, num = call.data.split('_')
#     answer = f'{call.message.from_user.first_name}, ты выбрал {call_type} и {call_but} и {num}'
    
#     await call.message.answer(answer)
#     await call.answer()

async def select_macbook(call: CallbackQuery, bot: Bot, callback_data: MacInfo):
    call_type, call_but, num = callback_data.call_type, callback_data.call_but, callback_data.num
    answer = f'{call.message.from_user.first_name}, ты выбрал {call_type} и {call_but} и {num}'
    
    await call.message.answer(answer)
    await call.answer()
