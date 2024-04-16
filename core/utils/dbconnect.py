import aiomysql

class Request:
    def __init__(self, connector: aiomysql.pool.Pool):
        self.connector = connector
        
    async def create_feedback(self, idTelegram):
        async with self.connector.cursor(aiomysql.DictCursor) as cur:
            query = f"INSERT INTO feedback (idTelegram, Type, Text, IsDone) " \
                    f"VALUES (1, 'test2', 'testtest2', false) " \
                    f"ON DUPLICATE KEY UPDATE " \
                    f"type = 'test2', " \
                    f"text = 'testtest2', " \
                    f"IsDone = false"
            await cur.execute(query)
            
    async def read_feedback(self):
        async with self.connector.cursor(aiomysql.DictCursor) as cur:
            query = f"SELECT * FROM feedback"
            await cur.execute(query)
            return await cur.fetchall()
        
    async def readone_feedback(self):
        async with self.connector.cursor(aiomysql.DictCursor) as cur:
            query = f"SELECT * FROM feedback"
            await cur.execute(query)
            return await cur.fetchone()
