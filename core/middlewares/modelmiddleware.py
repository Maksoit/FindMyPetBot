from aiogram.types import Message, TelegramObject
from aiogram import BaseMiddleware
from typing import Callable, Awaitable, Dict, Any
import aiomysql


class ModelMW(BaseMiddleware):
    def __init__(self, model):
        super().__init__()
        self.model = model

    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any]
        ) -> Any:
            data['model'] = self.model
            return await handler(event, data)
