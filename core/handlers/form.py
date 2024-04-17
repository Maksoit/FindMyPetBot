from aiogram import Bot
from aiogram.types import Message
from aiogram.fsm.context import FSMContext
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from core.utils.statesform import StepsForm
from core.handlers.apsched import send_message_middleware
from datetime import datetime, timedelta


async def get_form(message: Message, state: FSMContext):
    await message.answer(f'{message.from_user.first_name}, начнем заполнение анкеты. Введи свое имя:')
    await state.set_state(StepsForm.GET_NAME)    


async def get_name(message: Message, state: FSMContext):
    await message.answer(f'Твое имя:\r\n{message.text}\r\nТеперь введи фамилию:')
    await state.update_data(name=message.text)
    await state.set_state(StepsForm.GET_LAST_NAME)

async def get_last_name(message: Message, state: FSMContext):
    await message.answer(f'Твоя фамилия {message.text}.\r\nТеперь введи возраст:')
    await state.update_data(last_name=message.text)
    await state.set_state(StepsForm.GET_AGE)
    
async def get_age(message: Message, bot: Bot, state: FSMContext ): #apscheduler: AsyncIOScheduler
    await message.answer(f'Твой возраст: {message.text}\r\n')
    await state.update_data(age=message.text)
    context_data = await state.get_data()
    await message.answer(f'Сохранненные данные в машине состояний:\r\n{str(context_data)}')
    
    name = context_data.get('name')
    await message.answer(f'Ur name: {name}')
    await state.clear()
    
    test = input('KJkjkjkj')
    print(test)
    # apscheduler.add_job(send_message_middleware, trigger='date', run_date=datetime.now() + timedelta(seconds=13), kwargs={'chat_id':message.from_user.id} )