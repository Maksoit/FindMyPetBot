from aiogram import Bot, Dispatcher
from aiogram.types import Message
import asyncio
import logging #Блять что это???????????????????????????????????????????
import aiomysql

from aiogram.fsm.storage.redis import RedisStorage
# from apscheduler.jobstores.redis import RedisJobStore
# from apscheduler_di import ContextSchedulerDecorator

from core.handlers.basic import command_cancel, command_find, command_loss, get_description, get_location_find, get_location_loss, get_photo_find, get_photo_loss, get_start, get_hello, get_inline, test_handler
from core.handlers.callback import select_animal_find, select_find, select_loss, select_animal_loss
from core.filters.iscontact import IsTrueContact
from core.handlers.contact import get_fake_contact, get_true_contact_find, get_true_contact_loss
from core.keyboards.reply import get_reply_empty
from core.middlewares.vectoring import VectoringMW
from core.settings import Setting
from aiogram.filters import Command, CommandStart, callback_data
from aiogram import F
from core.utils.commands import set_commands
from core.utils.callbackdata import InlineInfo, MacInfo
from core.handlers.pay import order, pre_checkout_query, successful_payment, shipping_check
from core.middlewares.countermiddleware import CounterMiddleware
from core.middlewares.officehours import OfficeHoursMiddleware
from core.middlewares.apschedulermiddleware import SchedulerMiddleware
from core.middlewares.modelmiddleware import ModelMW
from core.middlewares.dbmiddleware import DBSession
from aiogram.utils.chat_action import ChatActionMiddleware
from model.Sasha import Vectorization
from model.Vova import init_model

from core.handlers import form
from core.utils.statesform import FindSteps, LossSteps, StepsForm
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from core.handlers import apsched
from datetime import datetime, timedelta


async def start_bot(bot: Bot):
    await set_commands(bot)
    await bot.send_message(Setting.bots.admin_id,
                           f"Bot has been launched at {datetime.now()}", reply_markup=get_reply_empty())
    print('Bot has been launched')

async def stop_bot(bot: Bot):
    await bot.send_message(Setting.bots.admin_id,
                           f"Bot has been stopped at {datetime.now()}")
    print("Bot has been stopped")

async def create_pool():
    return await aiomysql.create_pool(
        host = 'localhost', 
        port = Setting.database.port, 
        user= 'root',
        password=Setting.database.password,
        db='findmypetdb',
        autocommit=True,
        )

async def start():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - [%(levelname)s] - %(name)s - "
                        "(%(filename)s.%(funcName)s(%(lineno)d) - %(message)s")

    bot = Bot(token=Setting.bots.bot_token, parse_mode='HTML')

    pool_connect = await create_pool()
    
    model = init_model()
    # vectoring_model = Vectorization("", weight_path_resnet14=r"D:\Microsoft Visual Studio\Source\FindMyPetBot\model\resnet.pth", weight_path_simclr=r"D:\Microsoft Visual Studio\Source\FindMyPetBot\model\simclr.pth")

    storage = RedisStorage.from_url('redis://localhost:6379/0')

    dp = Dispatcher(storage=storage) 
    
    # jobstores = {
    #     'default': RedisJobStore(jobs_key='dispatched_trips_jobs',
    #                              run_times_key='dispatched_trips_running',
    #                              host='localhost',
    #                              db=2,
    #                              port=6379)
    #     }
    
    # scheduler = ContextSchedulerDecorator(AsyncIOScheduler(timezone="Europe/Moscow", jobstores=jobstores))
    # scheduler.ctx.add_instance(bot, declared_class=Bot)
    # scheduler.add_job(apsched.send_message_time, trigger='date', run_date= datetime.now() + timedelta(seconds=10))
    # scheduler.add_job(apsched.send_message_cron, trigger='cron', hour=datetime.now().hour, minute= datetime.now().minute + 1, start_date=datetime.now())
    # scheduler.add_job(apsched.send_message_interval, trigger='interval', seconds=60)
    # scheduler.start()


    # dp.message.middleware.register(CounterMiddleware())
    # dp.update.middleware.register(OfficeHoursMiddleware())
    # dp.update.middleware.register(SchedulerMiddleware(scheduler))

    dp.update.middleware.register(DBSession(pool_connect))
    dp.update.middleware.register(ModelMW(model))
    #dp.update.middleware.register(VectoringMW(vectoring_model))
    dp.message.middleware.register(ChatActionMiddleware())

    

    dp.message.register(get_start, Command(commands=['start', 'run']))  # CommandStart()
    dp.message.register(command_find, Command(commands='find'))
    dp.message.register(command_loss, Command(commands='loss'))
    dp.message.register(command_cancel, Command(commands='cancel'))

    dp.message.register(test_handler, Command(commands='test'))    

    dp.callback_query.register(select_loss, InlineInfo.filter(F.type == "loss"))
    dp.callback_query.register(select_find, InlineInfo.filter(F.type == "find"))
    dp.callback_query.register(select_animal_loss, InlineInfo.filter(), LossSteps.GET_ANIMAL)
    dp.callback_query.register(select_animal_find, InlineInfo.filter(), FindSteps.GET_ANIMAL)
    
    dp.message.register(get_location_loss, F.location, LossSteps.GET_LOCATION_CONTACT)
    dp.message.register(get_location_find, F.location, FindSteps.GET_LOCATION_CONTACT)
    dp.message.register(get_true_contact_loss, F.contact, IsTrueContact(), LossSteps.GET_LOCATION_CONTACT)
    dp.message.register(get_true_contact_find, F.contact, IsTrueContact(), FindSteps.GET_LOCATION_CONTACT)
    dp.message.register(get_description, LossSteps.GET_DESCRIPTION)
    dp.message.register(get_photo_loss, F.photo, LossSteps.GET_PHOTO)
    dp.message.register(get_photo_find, F.photo, FindSteps.GET_PHOTO)

    
    
    dp.message.register(get_fake_contact, F.contact)
    
    
    dp.startup.register(start_bot)
    dp.shutdown.register(stop_bot)
    


    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()

if __name__ == "__main__":
    asyncio.run(start())