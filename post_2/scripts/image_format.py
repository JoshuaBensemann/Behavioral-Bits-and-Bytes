from PIL import Image
from os import scandir, path, mkdir
import os

NEW_SIZE = (224, 224)
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
IMAGE_EXT = ['.jpg', '.jpeg', '.png', 'webp']
INPUT_DIR_A = path.join(DATA_DIR, 'model_input_a')
INPUT_DIR_B = path.join(DATA_DIR, 'model_input_b')

def resize_image(image_path, new_path, new_size=NEW_SIZE):
    image = Image.open(image_path)
    image.thumbnail(new_size)
    image.save(new_path)

if not path.exists(INPUT_DIR_A):
    mkdir(INPUT_DIR_A)

if not path.exists(INPUT_DIR_B):
    mkdir(INPUT_DIR_B)

monet_a = [file.name for file in scandir(f'{DATA_DIR}/Monet_A') if any(file.name.endswith(ext) for ext in IMAGE_EXT)] 
if not path.exists(f'{INPUT_DIR_A}/Monet'):
    mkdir(f'{INPUT_DIR_A}/Monet')

for image in monet_a:
    image_path = f'{DATA_DIR}/Monet_A/{image}'
    new_path = f'{INPUT_DIR_A}/Monet/{image}'
    if new_path.endswith('.webp'):
        new_path = new_path.replace('.webp', '.jpg')
    resize_image(image_path, new_path)

for file in scandir(f'{INPUT_DIR_A}/Monet'):
    print(file.name)

picasso_a = [file.name for file in scandir(f'{DATA_DIR}/Picasso_A') if any(file.name.endswith(ext) for ext in IMAGE_EXT)]
if not path.exists(f'{INPUT_DIR_A}/Picasso'):
    mkdir(f'{INPUT_DIR_A}/Picasso')

for image in picasso_a:
    image_path = f'{DATA_DIR}/Picasso_A/{image}'
    new_path = f'{INPUT_DIR_A}/Picasso/{image}'
    if new_path.endswith('.webp'):
        new_path = new_path.replace('.webp', '.jpg')
    resize_image(image_path, new_path)

for file in scandir(f'{INPUT_DIR_A}/Picasso'):
    print(file.name)

monet_b = [file.name for file in scandir(f'{DATA_DIR}/Monet_B') if any(file.name.endswith(ext) for ext in IMAGE_EXT)] 
if not path.exists(f'{INPUT_DIR_B}/Monet'):
    mkdir(f'{INPUT_DIR_B}/Monet')

for image in monet_b:
    image_path = f'{DATA_DIR}/Monet_B/{image}'
    new_path = f'{INPUT_DIR_B}/Monet/{image}'
    if new_path.endswith('.webp'):
        new_path = new_path.replace('.webp', '.jpg')
    resize_image(image_path, new_path)

for file in scandir(f'{INPUT_DIR_B}/Monet'):
    print(file.name)

picasso_b = [file.name for file in scandir(f'{DATA_DIR}/Picasso_B') if any(file.name.endswith(ext) for ext in IMAGE_EXT)]
if not path.exists(f'{INPUT_DIR_B}/Picasso'):
    mkdir(f'{INPUT_DIR_B}/Picasso')

for image in picasso_b:
    image_path = f'{DATA_DIR}/Picasso_B/{image}'
    new_path = f'{INPUT_DIR_B}/Picasso/{image}'
    if new_path.endswith('.webp'):
        new_path = new_path.replace('.webp', '.jpg')
    resize_image(image_path, new_path)

for file in scandir(f'{INPUT_DIR_B}/Picasso'):
    print(file.name)
