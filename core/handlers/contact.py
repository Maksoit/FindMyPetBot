from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from aiogram import Bot

from core.utils.statesform import LossSteps

async def get_true_contact(message: Message, bot: Bot, phone: str, state: FSMContext):
    await message.answer(f'Ты отправил <b>свой</b> контакт - {phone}')
    await state.update_data(phone = phone)
    context_data = await state.get_data()
    if (context_data.get('phone')):
        await state.set_state(LossSteps.GET_DESCRIPTION)
        await message.answer(f'Теперь в свободной форме опиши потерянное животное!')

async def get_fake_contact(message: Message, bot: Bot):
    await message.answer(f'Ты отправил <b>не свой</b> контакт - {message.contact.phone_number}')