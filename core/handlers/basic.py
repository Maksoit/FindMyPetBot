from aiogram import Bot, types
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
import json
from core.keyboards.reply import get_reply_empty, reply_keyboard, loc_tel_poll_keyboard, get_reply_keyboard
from core.keyboards.inline import get_inline_animal, select_macbook, get_inline_keyboard, get_inline_start
from core.utils.dbconnect import Request
from core.utils.statesform import FindSteps, LossSteps
from model.Sasha import Vectorization
from model.Vova import crop_image


async def test_handler(message: Message, bot: Bot, request: Request):
    await message.answer('TEST')
    
    

async def get_inline(message: Message, bot: Bot):
    await message.answer('Hello, its inline buttons', reply_markup=get_inline_keyboard())

async def get_start(message: Message, bot: Bot, request: Request):
    await message.answer("Привет!\nЯ - бот для пет-проекта \"Поиск домашних животных\", созданный REU Data Science Club. \nЯ создан, чтобы помогать людям находить их потерявшихся домашних животных" \
                         "\nЧтобы сообщить о своей пропаже отправь команду \"\\loss\"" \
                         "\nЧтобы сообщить о найденном животном на улице отправь команду \"\\find\" ", reply_markup=get_inline_start() )
  


async def get_location_loss(message: Message, bot: Bot, state: FSMContext):
    await message.answer(f'Ты отправил геолокацию!\r\a {message.location.latitude}\r\n{message.location.longitude}')
    await state.update_data(latitude = message.location.latitude)
    await state.update_data(longitude = message.location.longitude)
    context_data = await state.get_data()
    if (context_data.get('phone')):
        await state.set_state(LossSteps.GET_DESCRIPTION)
        await message.answer(f'Теперь в свободной форме опиши потерянное животное!', reply_markup=types.ReplyKeyboardRemove())
        
async def get_location_find(message: Message, bot: Bot, state: FSMContext):
    await message.answer(f'Ты отправил геолокацию!\r\n {message.location.latitude}\r\n{message.location.longitude}')
    await state.update_data(latitude = message.location.latitude)
    await state.update_data(longitude = message.location.longitude)
    context_data = await state.get_data()
    if (context_data.get('phone')):
        await state.set_state(FindSteps.GET_PHOTO)
        await message.answer(f'Теперь в свободной форме опиши потерянное животное!', reply_markup=types.ReplyKeyboardRemove())
    
        
async def get_photo_loss(message: Message, bot: Bot, state: FSMContext, request: Request, model, vectoring: Vectorization):
    # скачал фотку от потерявшего
    await message.answer(f'Отлично, ты отправил фото питомца! На этом регистрация запроса закончена', reply_markup=get_reply_empty())
    await message.answer(f'Начинается процесс поиска...', reply_markup=get_reply_empty())
    file = await bot.get_file(message.photo[-1].file_id)
    filename = 'photo' + str(file.file_id) + '.jpg'
    await bot.download_file(file.file_path, filename)
    
    # обрезал фотку согласно инфе от модели Вовы
    predict = model.predict(filename, confidence=40, overlap=30).json()
    print(predict)
    x, y, width, height = predict['predictions'][0]['x'], predict['predictions'][0]['y'], 256, 256 #predict['predictions'][0]['width'], predict['predictions'][0]['height']
    crop_image(filename, x - width/2, y - height/2, x + width/2, y + height/2)
    filename_croped = 'photo' + str(file.file_id) + "_croped" + '.jpg'

    # создал новую запись в БД потерянных репортов
    data = await state.get_data()
    # {'animal': 'cat', 'latitude': 51.758321, 'longitude': 28.623898, 'phone': '79778799694', 'description': 'Лалв'}
    result = await request.new_miss_rep(data["animal"], data["latitude"], data["longitude"], data["description"], filename_croped, message.from_user.id, data["phone"])
    print(result)
    
    # векторизовал обрезанную фотку и добавил в БД векторов
    vector = str(vectoring.vectorize_image(filename_croped)) # возвращается тензор, переведется ли он в str
    id_missing_report = await request.get_id_miss_report_by_filename(filename_croped)
    result = await request.new_vector(vector, missing_rep_id=id_missing_report)
    print(result)
    
    # получил все вектора, где не установлен миссинг_репорт и добавил их в массив vectors, посчитал среднекосинусное всех фото
    result = await request.get_vectors_street_pet() # тут нет нашей картинки
    vectors = [].append(vector) # все в str
    for i in result:
        vectors.append(i["Vector"])
    all_cos_dist = vectoring.average_cosine_similarity(vectors)
    
    # перебрал все пары и добавил в possible_variants вектора с косинусным с нашей кортинкой больше среднекосинусного
    possible_variants = []
    for i in range(1, len(vectors)):
        cos_dist = vectoring.cosine_distance(vectors[0], vectors[i])
        if (cos_dist >= all_cos_dist):
            possible_variants.append(vectors[i])
    
    # в possible_id_street_rep добавил id в БД "найденных на улице" со всех векторов, которые возможно подходят
    possible_id_street_rep = []
    for i in range(len(possible_variants)):
        id_street_rep = await request.get_id_street_rep_by_vector(possible_variants[i])
        possible_id_street_rep.append(id_street_rep)
        
    # собрал инфу из БД "найденных на улице" про все возможные id, получил массив словарей
    info_possible = []
    for i in range(len(possible_id_street_rep)):
        info = await request.get_info_by_street_rep_id(possible_id_street_rep[i])
        info_possible += info    
    
    await send_possible_info(message, bot, state, info_possible)

    await state.clear()
    
async def send_possible_info(message: Message, bot: Bot, state: FSMContext, info):
    pass

async def get_photo_find(message: Message, bot: Bot, state: FSMContext):
    await message.answer(f'Отлично, ты отправил фото животного! На этом регистрация запроса закончена', reply_markup=get_reply_empty())
    file = await bot.get_file(message.photo[-1].file_id)
    await bot.download_file(file.file_path, 'photo' + str(file.file_id) + '.jpg')
    await state.clear()
    

async def get_hello(message: Message, bot: Bot):
    await message.answer(f'И тебе привет!')
    json_str = json.dumps(message.dict(), default=str)
    print(json_str)

async def get_description(message: Message, bot: Bot, state: FSMContext):
    await state.update_data(description = message.text)
    await message.answer(f'Ты отправил описание! Теперь отправь фото своего питомца')
    await state.set_state(LossSteps.GET_PHOTO)
    
async def command_loss(message: Message, bot: Bot, state: FSMContext):
    await message.answer("Чтобы сообщить о потери нам понадобится некоторая информация.\nПожалуйста, выбери животное (нажми на кнопку)", reply_markup=get_inline_animal())
    await state.set_state(LossSteps.GET_ANIMAL)


async def command_find(message: Message, bot: Bot, state: FSMContext):
    await message.answer("Чтобы сообщить о находке нам понадобится некоторая информация.\nПожалуйста, выбери животное (нажми на кнопку)", reply_markup=get_inline_animal())
    await state.set_state(FindSteps.GET_ANIMAL)

async def command_cancel(message: Message, bot: Bot, state: FSMContext):
    await message.answer("Текущий этап завершен досрочно")
    await state.clear()

