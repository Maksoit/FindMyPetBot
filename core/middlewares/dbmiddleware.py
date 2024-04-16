from aiogram.types import Message, TelegramObject
from aiogram import BaseMiddleware
from typing import Callable, Awaitable, Dict, Any
import aiomysql
from core.utils.dbconnect import Request


class DBSession(BaseMiddleware):
    def __init__(self, connector: aiomysql.pool.Pool):
        super().__init__()
        self.connector = connector

    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any]
        ) -> Any:
            async with self.connector.acquire() as connect:
                data['request'] = Request(connect)
                return await handler(event, data)