from aiogram import Bot
from core.settings import Setting

async def send_message_time(bot: Bot):
    await bot.send_message(Setting.bots.admin_id, f'Это сообщение отправлено через несколько секунд после старта бота')
    
async def send_message_cron(bot: Bot):
    await bot.send_message(Setting.bots.admin_id, f'Это сообщение будет отправляться ежедневно в указанное время')
    
async def send_message_interval(bot: Bot):
    await bot.send_message(Setting.bots.admin_id, f'Это сообщение будет отправляться с интервалом в 1 минуту')
    
async def send_message_middleware(bot: Bot, chat_id: int):
    await bot.send_message(chat_id, f'Это сообщение отправлено с помощью сформированной через middleware задачи')
