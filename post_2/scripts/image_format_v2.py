import os
from os import mkdir, path, scandir

from PIL import Image, ImageOps

NEW_SIZE = (224, 224)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
IMAGE_EXT = [".jpg", ".jpeg", ".png", ".webp"]
TRAIN_DIR_A = path.join(DATA_DIR, "train_a")
TRAIN_DIR_B = path.join(DATA_DIR, "train_b")
TEST_4_DIR_A = path.join(DATA_DIR, "test_4_a")
TEST_4_DIR_B = path.join(DATA_DIR, "test_4_b")


def resize_image(image_path, new_path, new_size=NEW_SIZE):
    image = Image.open(image_path)
    image.thumbnail(new_size)
    delta_w = new_size[0] - image.size[0]
    delta_h = new_size[1] - image.size[1]
    padding = (
        delta_w // 2,
        delta_h // 2,
        delta_w - (delta_w // 2),
        delta_h - (delta_h // 2),
    )
    image = ImageOps.expand(image, padding)

    # Change the extension of the new path to .jpg
    new_path = path.splitext(new_path)[0] + ".jpg"
    image.save(new_path, "JPEG")


if not path.exists(TRAIN_DIR_A):
    mkdir(TRAIN_DIR_A)


if not path.exists(TRAIN_DIR_B):
    mkdir(TRAIN_DIR_B)


if not path.exists(TEST_4_DIR_A):
    mkdir(TEST_4_DIR_A)


if not path.exists(TEST_4_DIR_B):
    mkdir(TEST_4_DIR_B)

data_dir_content = [entry.name for entry in scandir(DATA_DIR) if entry.is_dir()]
print("Directories in DATA_DIR:", data_dir_content)

data_dir_content_A = [entry for entry in data_dir_content if entry.endswith("_A")]
data_dir_content_B = [entry for entry in data_dir_content if entry.endswith("_B")]

print("Directories ending with '_A':", data_dir_content_A)
print("Directories ending with '_B':", data_dir_content_B)

for dir_name in data_dir_content_A:
    source_dir = path.join(DATA_DIR, dir_name)
    if dir_name in ["Monet_A", "Picasso_A"]:
        target_dir = path.join(TRAIN_DIR_A, dir_name.replace("_A", ""))
    else:
        target_dir = path.join(TEST_4_DIR_A, dir_name.replace("_A", ""))

    if not path.exists(target_dir):
        mkdir(target_dir)

    for entry in scandir(source_dir):
        if entry.is_file() and path.splitext(entry.name)[1] in IMAGE_EXT:
            resize_image(
                path.join(source_dir, entry.name), path.join(target_dir, entry.name)
            )

# Equivalent for data_dir_content_B
for dir_name in data_dir_content_B:
    source_dir = path.join(DATA_DIR, dir_name)
    if dir_name in ["Monet_B", "Picasso_B"]:
        target_dir = path.join(TRAIN_DIR_B, dir_name.replace("_B", ""))
    else:
        target_dir = path.join(TEST_4_DIR_B, dir_name.replace("_B", ""))

    if not path.exists(target_dir):
        mkdir(target_dir)

    for entry in scandir(source_dir):
        if entry.is_file() and path.splitext(entry.name)[1] in IMAGE_EXT:
            resize_image(
                path.join(source_dir, entry.name), path.join(target_dir, entry.name)
            )
