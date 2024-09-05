from aiogram.types import Message, TelegramObject
from aiogram import BaseMiddleware
from typing import Callable, Awaitable, Dict, Any


class VectoringMW(BaseMiddleware):
    def __init__(self, vectoring):
        super().__init__()
        self.model = vectoring

    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any]
        ) -> Any:
            data['vectoring'] = self.model
            return await handler(event, data)