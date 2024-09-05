from aiogram import Bot
from aiogram.types import CallbackQuery
from core.utils.callbackdata import InlineInfo
from core.keyboards.inline import get_inline_animal, get_inline_geo_contact
from core.keyboards.reply import get_reply_geo
from aiogram.fsm.context import FSMContext
from core.utils.statesform import FindSteps, LossSteps

# async def select_macbook(call: CallbackQuery, bot: Bot):
#     call_type, call_but, num = call.data.split('_')
#     answer = f'{call.message.from_user.first_name}, ты выбрал {call_type} и {call_but} и {num}'
    
#     await call.message.answer(answer)
#     await call.answer()


    
async def select_loss(call: CallbackQuery, bot: Bot, callback_data: InlineInfo, state: FSMContext):
    await call.message.answer("Чтобы сообщить о потери нам понадобится некоторая информация.\nПожалуйста, выбери животное (нажми на кнопку)", reply_markup=get_inline_animal())
    await state.set_state(LossSteps.GET_ANIMAL)
    await call.answer()
    
async def select_animal_loss(call: CallbackQuery, bot: Bot, callback_data: InlineInfo, state: FSMContext):
    await call.message.answer(f"Ты выбрал {callback_data.animal}!\nТеперь нам понадобится информация о твоей геопозиции!\n Ты можешь отправить ее по кнопке с мобильного устройства" \
                              " или напечатать вручную в формате 'xx.xxx, yy.yyy'", reply_markup=get_reply_geo())
    await state.update_data(animal = callback_data.animal)
    await state.set_state(LossSteps.GET_LOCATION)
    await call.answer()
    

async def select_animal_find(call: CallbackQuery, bot: Bot, callback_data: InlineInfo, state: FSMContext):
    await call.message.answer(f"Ты выбрал {callback_data.animal}!\nТеперь нам понадобится информация о твоей геопозиции!\n Ты можешь отправить ее по кнопке с мобильного устройства" \
                              " или напечатать вручную в формате 'xx.xxx, yy.yyy'", reply_markup=get_reply_geo())
    await state.update_data(animal = callback_data.animal)
    await state.set_state(FindSteps.GET_LOCATION)
    await call.answer()
    

async def select_find(call: CallbackQuery, bot: Bot, callback_data: InlineInfo, state: FSMContext):
    await call.message.answer("Чтобы сообщить о находке нам понадобится некоторая информация.\nПожалуйста, выбери животное (нажми на кнопку)", reply_markup=get_inline_animal())
    await state.set_state(FindSteps.GET_ANIMAL)
    await call.answer()
    
