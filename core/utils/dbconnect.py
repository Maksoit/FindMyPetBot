import aiomysql
import datetime



class Request:
    def __init__(self, connector: aiomysql.pool.Pool):
        self.connector = connector
        
    async def create_own_SQL_request(self, data: str):
        async with self.connector.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(data)
            return await cur.fetchall()

    async def new_miss_rep(self, CatOrDog: str, f_coord: float, s_coord: float, desc: str, path: str, IdTg: int, Phone: int):
        date = str(datetime.date.today())
        CatOrDog = 0 if CatOrDog == "cat" else 1
        async with self.connector.cursor(aiomysql.DictCursor) as cur:
            query = f"insert into `missing reports` (`Cat or dog`, `First coord`, `Second coord`, Description, Path, Date, `Id tg`, `Phone number`)"\
                f" VALUES ({CatOrDog}, {f_coord}, {s_coord}, '{desc}', '{path}', '{date}', {IdTg}, {Phone});"
            await cur.execute(query)
            return await cur.fetchall()
        
    async def new_street_pet(self, CatOrDog: str, f_coord: float, s_coord: float, desc: str, path: str, IdTg: int, Phone: int):
        date = str(datetime.date.today())
        CatOrDog = 0 if CatOrDog == "cat" else 1
        async with self.connector.cursor(aiomysql.DictCursor) as cur:
            query = f"insert into `missing reports` (`Cat or dog`, `First coord`, `Second coord`, Description, Path, Date, `Id tg`, `Phone number`)"\
                f" VALUES ({CatOrDog}, {f_coord}, {s_coord}, '{desc}', '{path}', '{date}', {IdTg}, {Phone});"
            await cur.execute(query)
            return await cur.fetchall()
        
    async def new_vector(self, vector: str, street_pet_id = -1, missing_rep_id = -1):
        async with self.connector.cursor(aiomysql.DictCursor) as cur:
            if (street_pet_id == -1): 
                if (missing_rep_id == -1): query = f"insert into `vectors` (Vector) VALUES ('{str(vector)}');"
                else: query = f"insert into `vectors` (Vector, `Missing reports_id`) VALUES ('{str(vector)}', {missing_rep_id});"
            else:
                if (missing_rep_id == -1): query = f"insert into `vectors` (Vector, `Street pets_id`) VALUES ('{str(vector)}', {street_pet_id});"
                else: query = f"insert into `vectors` (Vector, `Street pets_id`, `Missing reports_id`) VALUES ('{str(vector)}', {street_pet_id}, {missing_rep_id});"
                
            await cur.execute(query)
            return await cur.fetchall()
        
    async def get_vectors_street_pet(self):
        async with self.connector.cursor(aiomysql.DictCursor) as cur:
            query = f"SELECT * FROM vectors where `Missing reports_id` is null;"
            await cur.execute(query)
            return await cur.fetchall()
        
    async def get_id_miss_report_by_filename(self, path):
        async with self.connector.cursor(aiomysql.DictCursor) as cur:
            query = f"SELECT * FROM `missing reports` where Path = '{path}';"
            await cur.execute(query)
            response = await cur.fetchall()
            return response[0]["idMissing reports"]
        
    async def get_id_street_rep_by_vector(self, vector: str) -> int:
        async with self.connector.cursor(aiomysql.DictCursor) as cur:
            query = f"SELECT * FROM vectors where vector = '{vector}';"
            await cur.execute(query)
            response = await cur.fetchall()
            return response[0]["Street pets_id"]
        
    async def get_info_by_street_rep_id(self, id):
        async with self.connector.cursor(aiomysql.DictCursor) as cur:
            query = f"SELECT * FROM `street pets` where `idStreet pets` = {id};"
            await cur.execute(query)
            return await cur.fetchall()
        
    