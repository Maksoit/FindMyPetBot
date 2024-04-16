from datetime import datetime
from aiogram.types import Message, TelegramObject
from aiogram import BaseMiddleware
from typing import Callable, Awaitable, Dict, Any

def office_hours() -> bool:
    return datetime.now().weekday() in (0, 1, 2, 3, 4, 5, 6) and datetime.now().hour in ([i for i in (range(0, 25))])


class OfficeHoursMiddleware(BaseMiddleware):
    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any]
        ) -> Any:
        if office_hours():
            return await handler(event, data)
        # else:
        #     await event.answer("Время работы бота: будни с 8 до 18")
            

