from aiogram import Bot
from aiogram.types import BotCommand, BotCommandScopeDefault

async def set_commands(bot: Bot):
    commands = [
        BotCommand(command='start', description='Запустить бота'),
        BotCommand(command='loss', description='Сообщить о пропаже'),
        BotCommand(command='find', description='Сообщить о находке'),
        BotCommand(command='help', description='Помощь и описание'),
        BotCommand(command='cancel', description='Сбросить')
    ]
    await bot.set_my_commands(commands, BotCommandScopeDefault())