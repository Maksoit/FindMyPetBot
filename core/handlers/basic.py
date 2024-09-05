import ast
from aiogram import Bot, types
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
import json

import tensorflow as tf
import torch
from torch import tensor
from core.keyboards.reply import get_reply_contact, get_reply_empty, reply_keyboard, loc_tel_poll_keyboard, get_reply_keyboard
from core.keyboards.inline import get_inline_animal, get_inline_start
from core.utils.dbconnect import Request
from core.utils.statesform import FindSteps, LossSteps
from model.Sasha import Vectorization
from model.Vova import comp_coords, crop_image

R = 4

async def test_handler(message: Message, bot: Bot, request: Request):
    result = await request.create_own_SQL_request("SELECT * FROM findmypetdb.vectors;")
    print("test ", result)
    
    result = await request.create_own_SQL_request("SELECT * FROM findmypetdb.`street pets`;")
    print("test ", result)
    
    print(str([1,2]))
    
    

async def get_inline(message: Message, bot: Bot):
    await message.answer('Hello, its inline buttons', reply_markup=get_inline_keyboard())

async def get_start(message: Message, bot: Bot, request: Request):
    await message.answer("Привет!\nЯ - бот для пет-проекта \"Поиск домашних животных\", созданный REU Data Science Club. \nЯ создан, чтобы помогать людям находить их потерявшихся домашних животных" \
                         "\nЧтобы сообщить о своей пропаже отправь команду \"\\loss\"" \
                         "\nЧтобы сообщить о найденном животном на улице отправь команду \"\\find\" ", reply_markup=get_inline_start() )
  


async def get_location_loss(message: Message, bot: Bot, state: FSMContext):
    await message.answer(f'Ты отправил геолокацию!\r\n {message.location.latitude}\r\n{message.location.longitude}')
    await state.update_data(latitude = message.location.latitude)
    await state.update_data(longitude = message.location.longitude)
    context_data = await state.get_data()
    if (context_data.get('phone')):
        await state.set_state(LossSteps.GET_DESCRIPTION)
        await message.answer(f'Теперь в свободной форме опиши потерянное животное!', reply_markup=types.ReplyKeyboardRemove())
    else:
        await state.set_state(LossSteps.GET_CONTACT)
        await message.answer(f'Теперь отправь свой номер!', reply_markup=get_reply_contact())
        
async def get_location_loss_message(message: Message, bot: Bot, state: FSMContext):
    print(message.text)
    try:
        coords = list(map(lambda x: float(x), message.text.replace(",", "").split()))
        print(coords)
        await message.answer(f'Ты отправил геолокацию!\r\n {coords[0]}\r\n{coords[1]}')
        await state.update_data(latitude = coords[0])
        await state.update_data(longitude = coords[1])
        context_data = await state.get_data()
        if (context_data.get('phone')):
            await state.set_state(LossSteps.GET_DESCRIPTION)
            await message.answer(f'Теперь в свободной форме опиши потерянное животное!', reply_markup=types.ReplyKeyboardRemove())
        else:
            await state.set_state(LossSteps.GET_CONTACT)
            await message.answer(f'Теперь отправь свой номер!', reply_markup=get_reply_contact())
    except (...):
        await message.answer(f'Неверный формат. Попробуй еще раз!')
    
    
        
async def get_location_find(message: Message, bot: Bot, state: FSMContext):
    await message.answer(f'Ты отправил геолокацию!\r\n {message.location.latitude}\r\n{message.location.longitude}')
    await state.update_data(latitude = message.location.latitude)
    await state.update_data(longitude = message.location.longitude)
    context_data = await state.get_data()
    if (context_data.get('phone')):
        await state.set_state(FindSteps.GET_DESCRIPTION)
        await message.answer(f'Теперь в свободной форме опиши потерянное животное!', reply_markup=types.ReplyKeyboardRemove())
    else:
        await state.set_state(FindSteps.GET_CONTACT)
        await message.answer(f'Теперь отправь свой номер!', reply_markup=get_reply_contact())
    

async def get_location_find_message(message: Message, bot: Bot, state: FSMContext):
    if all(isinstance(x, int) for x in message.text.replace(",", " ").replace(".", " ").split()):
        await message.answer(f'Ты отправил геолокацию!\r\n {message.split()[0]}\r\n{message.split()[1]}')
        await state.update_data(latitude = message.split()[0])
        await state.update_data(longitude = message.split()[1])
        context_data = await state.get_data()
        if (context_data.get('phone')):
            await state.set_state(FindSteps.GET_DESCRIPTION)
            await message.answer(f'Теперь в свободной форме опиши потерянное животное!', reply_markup=types.ReplyKeyboardRemove())
        else:
            await state.set_state(FindSteps.GET_CONTACT)
            await message.answer(f'Теперь отправь свой номер!', reply_markup=get_reply_contact())
        
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
    vector = vectoring.vectorize_image(filename_croped) 
    id_missing_report = await request.get_id_miss_report_by_filename(filename_croped)
    result = await request.new_vector(str(vector), missing_rep_id=id_missing_report)
    print(result)
    
    # получаю id уличных животных, найденных недалеко от места пропажи
    all_street_pets = await request.get_street_id_coords()
    near_street_pets_ids = list()
    for i in range(len(all_street_pets)):
        if comp_coords(data["latitude"], data["longitude"], all_street_pets[i]["First coord"], all_street_pets[i]["Second coord"], R):
            near_street_pets_ids.append(all_street_pets[i]["idStreet pets"])
    
    print(near_street_pets_ids)
    # получил все вектора, где не установлен миссинг_репорт и уличный id которых подходит по дистанции и добавил их в массив vectors_street_pets, посчитал среднекосинусное всех фото
    result = await request.get_vectors_street_pet(near_street_pets_ids) # тут нет нашей картинки
    print("# получил все вектора, где не установлен миссинг_репорт и подходит радиус", result)
    vectors_street_pets = vector
    for i in result:
        vector_temp = eval(i["Vector"])
        vectors_street_pets = torch.cat((vectors_street_pets, vector_temp))
    print("получил все вектора, где не установлен миссинг_репорт ", vectors_street_pets)
    all_cos_dist = vectoring.average_cosine_similarity(vectors_street_pets)
    
    # перебрал все пары и добавил в possible_variants вектора с косинусным с нашей кортинкой больше среднекосинусного
    possible_variants = list()
    print("len(vectors_street_pets) ", len(vectors_street_pets))
    for i in range(1, len(vectors_street_pets)):
        cos_dist = vectoring.cosine_distance(vectors_street_pets[0], vectors_street_pets[i])
        print("COSDIST ", vectors_street_pets[0], vectors_street_pets[i], cos_dist)
        if (cos_dist >= all_cos_dist):
            possible_variants.append(vectors_street_pets[i])
    print("possible_variants", possible_variants)
    
    # в possible_id_street_rep добавил id в БД "найденных на улице" со всех векторов, которые возможно подходят
    possible_id_street_rep = list()
    for i in range(len(possible_variants)):
        id_street_rep = await request.get_id_street_rep_by_vector(str(possible_variants[i]))
        possible_id_street_rep.append(id_street_rep)
        
    # собрал инфу из БД "найденных на улице" про все возможные id, получил массив словарей
    info_possible = list()
    for i in range(len(possible_id_street_rep)):
        info = await request.get_info_by_street_rep_id(int(possible_id_street_rep[i]))
        info_possible += info    
    
    await send_possible_info(message, bot, state, info_possible)

    await state.clear()
    
async def send_possible_info(message: Message, bot: Bot, state: FSMContext, info):
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    if (info):
        await message.answer(f'Найдены сообщения об уличных животных, похожих на Вашего питомца!\n Свяжитесь с владельцами по номерам: {info[0]["phone"]}', reply_markup=get_reply_empty())
    else:
        await message.answer(f'К сожалению, мы не нашли Вашего питомца.\n Мы продолжим поиски и сообщим Вам об изменениях', reply_markup=get_reply_empty())
    

async def get_photo_find(message: Message, bot: Bot, state: FSMContext, request: Request, model, vectoring: Vectorization):
    # скачал фотку от потерявшего
    await message.answer(f'Отлично, ты отправил фото животного! На этом регистрация запроса закончена, спасибо', reply_markup=get_reply_empty())
    file = await bot.get_file(message.photo[-1].file_id)
    filename = 'photo' + str(file.file_id) + '.jpg'
    await bot.download_file(file.file_path, filename)
    
    # обрезал фотку согласно инфе от модели Вовы
    predict = model.predict(filename, confidence=40, overlap=30).json()
    print(predict)
    x, y, width, height = predict['predictions'][0]['x'], predict['predictions'][0]['y'], 256, 256 #predict['predictions'][0]['width'], predict['predictions'][0]['height']
    crop_image(filename, x - width/2, y - height/2, x + width/2, y + height/2)
    filename_croped = 'photo' + str(file.file_id) + "_croped" + '.jpg'

    # создал новую запись в БД найденных на улице
    data = await state.get_data()
    print("# создал новую запись в БД найденных на улице - data", data)
    # {'animal': 'cat', 'latitude': 51.758321, 'longitude': 28.623898, 'phone': '79778799694', 'description': 'Лалв'}
    result = await request.new_street_pet(data["animal"], data["latitude"], data["longitude"], data["description"], filename_croped, message.from_user.id, data["phone"])
    print("# создал новую запись в БД найденных на улице", result)
    
    # векторизовал обрезанную фотку и добавил в БД векторов
    vector = str(vectoring.vectorize_image(filename_croped)) # возвращается тензор, переведется ли он в str
    id_find_report = await request.get_id_find_report_by_filename(filename_croped)
    result = await request.new_vector(vector, street_pet_id=id_find_report)
    print("# векторизовал обрезанную фотку и добавил в БД векторов", result)

    await state.clear()
    

async def get_description_LOSS(message: Message, bot: Bot, state: FSMContext):
    await state.update_data(description = message.text)
    await message.answer(f'Ты отправил описание! Теперь отправь фото своего питомца')
    await state.set_state(LossSteps.GET_PHOTO)
    
async def get_description_FIND(message: Message, bot: Bot, state: FSMContext):
    await state.update_data(description = message.text)
    await message.answer(f'Ты отправил описание! Теперь отправь фото своего питомца')
    await state.set_state(FindSteps.GET_PHOTO)
    
async def command_loss(message: Message, bot: Bot, state: FSMContext):
    await message.answer("Чтобы сообщить о потери нам понадобится некоторая информация.\nПожалуйста, выбери животное (нажми на кнопку)", reply_markup=get_inline_animal())
    await state.set_state(LossSteps.GET_ANIMAL)


async def command_find(message: Message, bot: Bot, state: FSMContext):
    await message.answer("Чтобы сообщить о находке нам понадобится некоторая информация.\nПожалуйста, выбери животное (нажми на кнопку)", reply_markup=get_inline_animal())
    await state.set_state(FindSteps.GET_ANIMAL)

async def command_cancel(message: Message, bot: Bot, state: FSMContext):
    await message.answer("Текущий этап завершен досрочно")
    await state.clear()

