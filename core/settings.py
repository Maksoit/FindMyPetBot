from environs import Env
from dataclasses import dataclass


@dataclass
class Bots:
    bot_token: str
    admin_id: int
    
@dataclass
class DB:
    port: int
    password: str
    
@dataclass
class ML:
    VOVA_API_KEY: str

@dataclass
class Settings:
    bots: Bots
    database: DB
    ml: ML

def get_settings(path: str):
    env = Env()
    env.read_env(path)

    return Settings(bots=Bots(bot_token=env.str("DEV_TOKEN"), admin_id=env.int("ADMIN_ID")), database=DB(port=env.int("DB_PORT"), password=env.str("DB_PASSWORD")), ml = ML(VOVA_API_KEY=env.str("VOVA_API_KEY")))

Setting = get_settings('Confidential')