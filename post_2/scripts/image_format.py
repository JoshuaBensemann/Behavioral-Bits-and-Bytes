
from PIL import Image
from os import scandir, path, mkdir

NEW_SIZE = (224, 224)
DATA_DIR = '../data'
IMAGE_EXT = ['.jpg', '.jpeg', '.png', 'webp']
INPUT_DIR = path.join(DATA_DIR, 'model_input')

def resize_image(image_path, new_path, new_size=NEW_SIZE):
    image = Image.open(image_path)
    image.thumbnail(new_size)
    image.save(new_path)

if not path.exists(INPUT_DIR):
    mkdir(INPUT_DIR)

monet_a = [file.name for file in scandir(f'{DATA_DIR}/Monet_A') if any(file.name.endswith(ext) for ext in IMAGE_EXT)] 
if not path.exists(f'{INPUT_DIR}/Monet_A'):
    mkdir(f'{INPUT_DIR}/Monet_A')

for image in monet_a:
    image_path = f'{DATA_DIR}/Monet_A/{image}'
    new_path = f'{INPUT_DIR}/Monet_A/{image}'
    if new_path.endswith('.webp'):
        new_path = new_path.replace('.webp', '.jpg')
 
    resize_image(image_path, new_path)

for file in scandir(f'{INPUT_DIR}/Monet_A'):
    print(file.name)

monet_b = [file.name for file in scandir(f'{DATA_DIR}/Monet_B') if any(file.name.endswith(ext) for ext in IMAGE_EXT)] 
if not path.exists(f'{INPUT_DIR}/Monet_B'):
    mkdir(f'{INPUT_DIR}/Monet_B')

for image in monet_b:
    image_path = f'{DATA_DIR}/Monet_B/{image}'
    new_path = f'{INPUT_DIR}/Monet_B/{image}'
    if new_path.endswith('.webp'):
        new_path = new_path.replace('.webp', '.jpg')
 
    resize_image(image_path, new_path)

for file in scandir(f'{INPUT_DIR}/Monet_B'):
    print(file.name)

picasso_a = [file.name for file in scandir(f'{DATA_DIR}/Picasso_A') if any(file.name.endswith(ext) for ext in IMAGE_EXT)]
if not path.exists(f'{INPUT_DIR}/Picasso_A'):
    mkdir(f'{INPUT_DIR}/Picasso_A')

for image in picasso_a:
    image_path = f'{DATA_DIR}/Picasso_A/{image}'
    new_path = f'{INPUT_DIR}/Picasso_A/{image}'
    if new_path.endswith('.webp'):
        new_path = new_path.replace('.webp', '.jpg')
 
    resize_image(image_path, new_path)

for file in scandir(f'{INPUT_DIR}/Picasso_A'):
    print(file.name)

picasso_b = [file.name for file in scandir(f'{DATA_DIR}/Picasso_B') if any(file.name.endswith(ext) for ext in IMAGE_EXT)]
if not path.exists(f'{INPUT_DIR}/Picasso_B'):
    mkdir(f'{INPUT_DIR}/Picasso_B')

for image in picasso_b:
    image_path = f'{DATA_DIR}/Picasso_B/{image}'
    new_path = f'{INPUT_DIR}/Picasso_B/{image}'
    if new_path.endswith('.webp'):
        new_path = new_path.replace('.webp', '.jpg')
 
    resize_image(image_path, new_path)

for file in scandir(f'{INPUT_DIR}/Picasso_B'):
    print(file.name)
