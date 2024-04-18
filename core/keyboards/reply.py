from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, KeyboardButtonPollType
from aiogram.utils.keyboard import ReplyKeyboardBuilder

reply_keyboard = ReplyKeyboardMarkup(keyboard=[
    [
        KeyboardButton(text='Ряд 1. Кнопка 1'),
        KeyboardButton(text='Ряд 1. Кнопка 2'),
        KeyboardButton(text='Ряд 1. Кнопка 3')
    ],
    [
        KeyboardButton(text='Ряд 2. Кнопка 1'),
        KeyboardButton(text='Ряд 2. Кнопка 2'),
        KeyboardButton(text='Ряд 2. Кнопка 3'),
        KeyboardButton(text='Ряд 2. Кнопка 4')
    ],
    [
        KeyboardButton(text='Ряд 3. Кнопка 1'),
        KeyboardButton(text='Ряд 3. Кнопка 2')
    ]
], resize_keyboard=True, one_time_keyboard=True, input_field_placeholder='ЛУЛУЛКЛ', selective=True)

loc_tel_poll_keyboard = ReplyKeyboardMarkup(keyboard=[
    [
        KeyboardButton(text='Отправить геолокацию', request_location=True)
    ],
    [
        KeyboardButton(text='Отправить контакт', request_contact=True)
    ],
    [
        KeyboardButton(text='Создать викторину', request_poll= KeyboardButtonPollType())
    ]
], resize_keyboard=True, one_time_keyboard=False, input_field_placeholder='Отправь локу, тел или создай викт')


def get_reply_keyboard():
    keyboard_builder = ReplyKeyboardBuilder()

    keyboard_builder.button(text='Кнопка 1')
    keyboard_builder.button(text='Кнопка 2')
    keyboard_builder.button(text='Кнопка 3')
    keyboard_builder.button(text='Отправить гео', request_location=True)
    keyboard_builder.button(text='Отправить контакт', request_contact=True)
    keyboard_builder.button(text='Создать викторину', request_poll=KeyboardButtonPollType())
    keyboard_builder.adjust(3, 2, 1)
    return keyboard_builder.as_markup(resize_keyboard = True, one_time_keyboard= True, input_field_placeholder='Отправь локу, тел или создай викт')

def get_reply_geo_contact():
    keyboard_builder = ReplyKeyboardBuilder()

    keyboard_builder.button(text='Отправить гео', request_location=True)
    keyboard_builder.button(text='Отправить контакт', request_contact=True)
    keyboard_builder.adjust(1, 1)
    return keyboard_builder.as_markup(resize_keyboard = True, one_time_keyboard= False, input_field_placeholder='Отправь локацию, а затем свой контакт')

def get_reply_empty():
    keyboard_builder = ReplyKeyboardBuilder()
    return keyboard_builder.as_markup(resize_keyboard = True, one_time_keyboard= True)
