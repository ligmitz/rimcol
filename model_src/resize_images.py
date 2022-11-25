import os

from PIL import Image

path = "model_src/train_raw/"
save_path = "model_src/train/"

if __name__ == "__main__":
    for image_name in os.listdir(path):
        image = Image.open(f"{path}{image_name}")

        image = image.convert('RGBA')

        new_image = image.resize((256, 256))

        new_image.save(f"{save_path}{image_name.split('.')[0]}.png", format="png")
