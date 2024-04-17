from aiogram import Bot
from aiogram.types import CallbackQuery
from core.utils.callbackdata import MacInfo, InlineInfo

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
    
async def select_loss(call: CallbackQuery, bot: Bot, callback_data: InlineInfo):
    await call.message.answer("Чтобы сообщить о потери нам понадобится некоторая информация.")
    await call.answer()

async def select_find(call: CallbackQuery, bot: Bot, callback_data: InlineInfo):
    await call.message.answer("Чтобы сообщить о находке нам понадобится некоторая информация.")
    await call.answer()
    
