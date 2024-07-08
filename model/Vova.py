from roboflow import Roboflow

from asyncio.windows_events import NULL
from math import sin, cos, sqrt, atan2, radians
from PIL import Image

from core.settings import Setting

def init_model():
    rf = Roboflow(api_key=Setting.ml.VOVA_API_KEY)
    project = rf.workspace("dogsdetection-lv3ut").project("dogsdetectionfull")
    model = project.version(3).model
    return model

#test = model.predict("D:\Microsoft Visual Studio\Source\FindMyPetBot\photoAgACAgIAAxkBAAIB92Z8jKekR4pLFcDKWUPSXXsDU658AAJG4TEbYqnpS21ZXt1nOOzfAQADAgADeQADNQQ.jpg", confidence=40, overlap=30).json()
#print(test['predictions'][0]['x'])


def comp_coords(x_c, y_c, x_p, y_p, r):
    R = 6373.0 # Approximate radius of earth in km
    try:
        lat1 = radians(x_c)
        lon1 = radians(y_c)
        lat2 = radians(x_p)
        lon2 = radians(y_p)

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = R * c

        if (distance <= r): return 1
        else: return 0
    except Exception as expt:
        print("Error in comp_coords():\n" + str(expt))
        return -1

def crop_image(path, left, up, right, down, exit_path = NULL):
    try:
        if (not exit_path): Image.open(path).crop((left, up, right, down)).save(path.split('.')[0] + "_croped." + path.split('.')[1], quality=100) 
        else: Image.open(path).crop((left, up, right, down)).save(exit_path, quality=100)
    
    except FileNotFoundError as fnfe:
        print("Error in crop_image() with input file:\n" + str(fnfe))
        return NULL
    except ValueError as ve:
        print("Error in crop_image() with output file:\n" + str(ve))
        return NULL
    except Exception as expt:
        print("Error in crop_image():\n" + str(expt))
        return NULL
    
        


