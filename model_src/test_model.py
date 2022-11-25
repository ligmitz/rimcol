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
        # import matplotlib.pyplot as plt
        import numpy as np
        import tensorflow as tf
        from keras import models
        from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
        from keras.preprocessing.image import img_to_array, load_img
        from skimage.color import lab2rgb, rgb2lab, gray2rgb, rgb2gray
        from skimage.transform import resize
        from imageio import imwrite

        tf.compat.v1.disable_eager_execution()

        model = models.load_model(f"model_src/models/{name}")
        inception = InceptionResNetV2(weights="imagenet", include_top=True)
        inception.graph = tf.compat.v1.get_default_graph()

        images = []

        for image_name in listdir(test_input):
            image = img_to_array(load_img(test_input + image_name))
            images.append(image)

        images_arr = (1.0 / 255) * np.array(images, dtype=float)
        lab_images = rgb2lab(images_arr)
        grayed_rgb_images = gray2rgb(rgb2gray(images_arr))

        sample_grayscaled_rgb_resized = []

        for img in grayed_rgb_images:
            img = resize(img, (299, 299, 3), mode="constant")
            sample_grayscaled_rgb_resized.append(img)
        
        sample_grayscaled_rgb_resized = preprocess_input(np.array(sample_grayscaled_rgb_resized))

        with inception.graph.as_default():
            embedding = inception.predict(sample_grayscaled_rgb_resized)

        intensity_layer = lab_images[:, :, :, 0]
        intensity_layer = intensity_layer.reshape(intensity_layer.shape + (1,))

        output_layers = model.predict([intensity_layer, embedding])
        output_layers = output_layers * 128

        for output_index in range(len(output_layers)):
            canvas = np.zeros((256, 256, 3))
            canvas[:, :, 0] = intensity_layer[output_index][:, :, 0]
            canvas[:, :, 1:] = output_layers[output_index]
            canvas_rgb = lab2rgb(canvas) * 255
            # print(canvas_rgb.astype("uint8"))
            imwrite(f"{test_output}image-{str(output_index)}.png", canvas_rgb.astype("uint8"), "png")

            # plt.imshow(canvas_rgb.astype("uint8"))
            # plt.show()

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
