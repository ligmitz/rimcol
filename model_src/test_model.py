#!/usr/bin/env python3

import argparse
import sys
from os import listdir, mkdir, path


def test(name: str, test_input: str, test_output: str) -> None:
    if not path.isdir(f"model_src/models/{name}"):
        response = input(
            f"{name} model does not exist, please generate the model first"
        )
        sys.exit(1)

    if not path.isdir(test_input):
        print("Test input directory does not exist!")
        sys.exit(1)

    if not path.isdir(test_output):
        mkdir(test_output)

    try:
        import numpy as np
        from keras import models
        from keras.preprocessing.image import img_to_array, load_img
        from skimage.color import lab2rgb, rgb2lab
        from skimage.io import imsave

        model = models.load_model(f"model_src/models/{name}")

        images = []
        for image_name in listdir(test_input):
            image = img_to_array(load_img(test_input + image_name))
            images.append(image)

        images_arr = np.array(images, dtype=float)
        lab_images = rgb2lab((1.0 / 255) * images_arr)
        input_layers = lab_images[:, :, :, 0]
        input_layers = input_layers.reshape(input_layers.shape + (1,))

        output_layers = model.predict(input_layers)
        output_layers = output_layers * 128

        for output_index in range(len(output_layers)):
            canvas = np.zeros((256, 256, 3))
            canvas[:, :, 0] = input_layers[output_index][:, :, 0]
            canvas[:, :, 1:] = output_layers[output_index]
            imsave(f"{test_output}image-{str(output_index)}.png", lab2rgb(canvas))
    except Exception as e:
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Name of the model")
    parser.add_argument(
        "--test-input-dir",
        help="The directory containing input test images",
        default="model_src/test_input/",
    )
    parser.add_argument(
        "--test-output-dir",
        help="The directory containing test images output",
        default="model_src/test_output/",
    )

    args = parser.parse_args()
    test(
        args.name,
        args.test_input_dir,
        args.test_output_dir,
    )
