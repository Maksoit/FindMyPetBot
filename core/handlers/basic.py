from aiogram import Bot
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
import json
from core.keyboards.reply import get_reply_empty, reply_keyboard, loc_tel_poll_keyboard, get_reply_keyboard
from core.keyboards.inline import select_macbook, get_inline_keyboard, get_inline_start
from core.utils.dbconnect import Request
from core.utils.statesform import LossSteps


async def get_inline(message: Message, bot: Bot):
    await message.answer('Hello, its inline buttons', reply_markup=get_inline_keyboard())

async def get_start(message: Message, bot: Bot, request: Request):
    # await request.create_feedback(message.from_user.id)

    # await message.answer(f'Сообщение #{counter}')
    # await bot.send_message(message.from_user.id, f"{message.from_user.first_name}, а ты знал, что <b>Это send_message</b> ")
    await message.answer("Привет!\nЯ - бот для пет-проекта \"Поиск домашних животных\", созданный REU Data Science Club. \nЯ создан, чтобы помогать людям находить их потерявшихся домашних животных" \
                         "\nЧтобы сообщить о своей пропаже отправь команду \"\\loss\"" \
                         "\nЧтобы сообщить о найденном животном на улице отправь команду \"\\find\" ", reply_markup=get_inline_start() )
    # await message.reply(f"Это message.reply")
    # await bot.send_message(message.from_user.id,
     #                      f"Твой id: {message.from_user.id}", reply_markup=get_reply_keyboard())


async def get_location(message: Message, bot: Bot, state: FSMContext):
    await message.answer(f'Ты отправил геолокацию!\r\a {message.location.latitude}\r\n{message.location.longitude}')
    await state.update_data(latitude = message.location.latitude)
    await state.update_data(longitude = message.location.longitude)
    context_data = await state.get_data()
    if (context_data.get('phone')):
        await state.set_state(LossSteps.GET_DESCRIPTION)
        await message.answer(f'Теперь в свободной форме опиши потерянное животное!')
    

async def get_photo(message: Message, bot: Bot, state: FSMContext):
    await message.answer(f'Отлично, ты отправил фото питомца! На этом регистрация запроса закончена', reply_markup=get_reply_empty())
    file = await bot.get_file(message.photo[-1].file_id)
    await bot.download_file(file.file_path, 'photo' + str(file.file_id) + '.jpg')
    await state.clear()
    

async def get_hello(message: Message, bot: Bot):
    await message.answer(f'И тебе привет!')
    json_str = json.dumps(message.dict(), default=str)
    print(json_str)

async def get_description(message: Message, bot: Bot, state: FSMContext):
    await state.update_data(description = message.text)
    await message.answer(f'Ты отправил описание! Теперь отправь фото своего питомца')
    await state.set_state(LossSteps.GET_PHOTO)



