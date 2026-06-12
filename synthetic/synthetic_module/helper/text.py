from .seed import use_seed
from numpy.random import choice
import numpy as np
from .resources import (
    DATABASE,
    TEXT_RESRC_NAME,
    FONT_RESRC_NAME,
    CHINESE_TEXT_RESRC_NAME,
    ARABIC_TEXT_RESRC_NAME,
    GREEK_TEXT_RESRC_NAME,
    HEBREW_TEXT_RESRC_NAME,
)
from random import randint
from PIL import ImageFont

FONT_TYPES = ["arabic", "chinese", "handwritten", "normal", "number", "greek", "hebrew"]
TEXT_FONT_TYPE_RATIO = {
    "arabic": 0.4,
    "chinese": 0.15,
    "handwritten": 0.15,
    "normal": 0.1,
    "number": 0.05,
    "greek": 0.1,
    "hebrew": 0.05,
}

LATIN_FONT_TYPES = ["handwritten", "normal", "foreign_like"]
LATIN_TEXT_FONT_TYPE_RATIO = {
    "handwritten": 0.2,
    "normal": 0.6,
    "foreign_like": 0.2,
}

MIN_NB_CHARACTERS = 100
MIN_IMG_DIMENSION = 200
TEXT_BASELINE_HEIGHT = 5
TEXT_BBOX_FREQ = 0

TEXT_BBOX_BORDER_WIDTH_RANGE = (1, 6)
TEXT_BBOX_PADDING_RANGE = (0, 20)
TEXT_COLORED_FREQ = 0.5
TEXT_JUSTIFIED_PARAGRAPH_FREQ = 0.7
TEXT_ROTATION_ANGLE_RANGE = (-60, 60)
TEXT_TIGHT_PARAGRAPH_FREQ = 0.5
TEXT_TITLE_UPPERCASE_RATIO = 0.5
TEXT_TITLE_UNILINE_RATIO = 0.25
TEXT_UNDERLINED_FREQ = 0


TEXT_UNDERLINED_PADDING_RANGE = (0, 4)
FONT_SIZE_RANGE = (15, 30)


@use_seed()
def get_random_font(is_latin):
    if is_latin:
        font_type = choice(
            list(LATIN_TEXT_FONT_TYPE_RATIO.keys()),
            p=list(LATIN_TEXT_FONT_TYPE_RATIO.values()),
        )
    else:
        font_type = choice(
            list(TEXT_FONT_TYPE_RATIO.keys()), p=list(TEXT_FONT_TYPE_RATIO.values())
        )
    return choice(DATABASE[FONT_RESRC_NAME][font_type])

def get_dictionary(parameters, height):
    is_latin = parameters["is_latin"]
    min_fs, max_fs = FONT_SIZE_RANGE
    font_path = parameters.get("font_path") or get_random_font(is_latin)

    rescaled_height = (height * 2) // 3  
    actual_max_fs = min(rescaled_height, max_fs)
    if min_fs < actual_max_fs:
        font_size = randint(min_fs, actual_max_fs)
    else:
        font_size = actual_max_fs
    font = ImageFont.truetype(font_path, size=font_size)
    if "text" in parameters:
        text = parameters["text"]
    else:
        n_char = 0
        if "chinese" in font_path:
            text_resource = DATABASE[CHINESE_TEXT_RESRC_NAME]
        elif "arabic" in font_path:
            text_resource = DATABASE[ARABIC_TEXT_RESRC_NAME]


        elif 'greek' in font_path:
            text_resource = DATABASE[GREEK_TEXT_RESRC_NAME]
        elif 'hebrew' in font_path:
            text_resource = DATABASE[HEBREW_TEXT_RESRC_NAME]
        elif 'number' in font_path:
            ## random numbers
            text = np.random.randint(0, 9)
            dictionary = list(str(text))
        else:
            text_resource = DATABASE[TEXT_RESRC_NAME]
        
        while n_char <= 100:
            text_path = choice(text_resource)
            with open(text_path) as f:
                text = f.read().rstrip("\n")
            n_char = len(text)
    if "chinese" in font_path:
        dictionary = list(text)
    else:
        dictionary = text.split("\n")
    return dictionary, font_path, font_size