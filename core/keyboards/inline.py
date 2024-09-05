from tkinter import N
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, keyboard_button
from aiogram.utils.keyboard import InlineKeyboardBuilder
from core.utils.callbackdata import InlineInfo



def get_inline_start():
    keyboard_builder = InlineKeyboardBuilder()
    keyboard_builder.button(text='Отправить запрос о потере своего животного', callback_data=InlineInfo(type='loss', IsLocated=False, IsContact=False, animal='') )
    keyboard_builder.button(text='Отправить запрос о находке животного на улице', callback_data=InlineInfo(type='find', IsLocated=False, IsContact=False, animal=''))
    
    keyboard_builder.adjust(1, 1)
    return keyboard_builder.as_markup()

def get_inline_animal():
    keyboard_builder = InlineKeyboardBuilder()
    keyboard_builder.button(text='Кот', callback_data=InlineInfo(type='', IsLocated=False, IsContact=False, animal='cat'))
    keyboard_builder.button(text='Собака', callback_data=InlineInfo(type='', IsLocated=False, IsContact=False, animal='dog'))
    
    keyboard_builder.adjust(2)
    return keyboard_builder.as_markup()

def get_inline_geo_contact():
    keyboard_builder = InlineKeyboardBuilder()
    keyboard_builder.button(text='Отправить геолокацию', callback_data=InlineInfo(IsLocated=True, IsContact=False, type='', animal=''), request_location = True)
    keyboard_builder.button(text='Отправить свой контакт', callback_data=InlineInfo(IsLocated=False, IsContact=True, type='', animal=''), request_contact = True)
    
    keyboard_builder.adjust(1, 1)
    return keyboard_builder.as_markup()