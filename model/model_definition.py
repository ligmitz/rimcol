import os
import random

import numpy as np
from keras.layers import Conv2D, InputLayer, UpSampling2D
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img
from numpy import ndarray
from skimage.color import lab2rgb, rgb2lab
from skimage.io import imsave


class Model:
    def __init__(self, test_dir: str, train_dir: str, model_name: str) -> None:
        self.model = Sequential()
        self.model_name = model_name
        self.test_dir = test_dir
        self.train_dir = train_dir

    def load_image(self, img_name: str) -> ndarray:
        image = img_to_array(load_img(img_name))
        image = np.array(image, dtype=float)

        return image

    def restructure_image_array(self, image: ndarray) -> tuple[ndarray, ndarray]:
        lab_image = rgb2lab((1.0 / 255) * image)
        intensity_layer = lab_image[:, :, 0]
        color_layers = lab_image[:, :, 1:]

        color_layers = color_layers / 128
        intensity_layer = intensity_layer.reshape(1, 400, 400, 1)
        color_layers = color_layers.reshape(1, 400, 400, 2)

        return (intensity_layer, color_layers)

    def create_nn(self) -> None:
        self.model.add(InputLayer(input_shape=(None, None, 1)))
        self.model.add(Conv2D(8, (3, 3), activation="relu", padding="same", strides=2))
        self.model.add(Conv2D(8, (3, 3), activation="relu", padding="same"))
        self.model.add(Conv2D(16, (3, 3), activation="relu", padding="same"))
        self.model.add(Conv2D(16, (3, 3), activation="relu", padding="same", strides=2))
        self.model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
        self.model.add(Conv2D(32, (3, 3), activation="relu", padding="same", strides=2))
        self.model.add(UpSampling2D((2, 2)))
        self.model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
        self.model.add(UpSampling2D((2, 2)))
        self.model.add(Conv2D(16, (3, 3), activation="relu", padding="same"))
        self.model.add(UpSampling2D((2, 2)))
        self.model.add(Conv2D(2, (3, 3), activation="tanh", padding="same"))

    def test_image(self, image: ndarray) -> None:
        output_image = self.model.predict(image)
        output_image = output_image * 128

        canvas = np.zeros((400, 400, 3))
        canvas[:, :, 0] = image[0][:, :, 0]
        canvas[:, :, 1:] = output_image[0]
        imsave("model/test_output/people.png", lab2rgb(canvas))

    def run(self) -> None:
        image_name = "model/test_input/people.jpg"
        image = self.load_image(image_name)
        layer1, layer2 = self.restructure_image_array(image)

        self.create_nn()
        self.model.compile(optimizer="rmsprop", loss="mse")

        self.model.fit(x=layer1, y=layer2, batch_size=1, epochs=1000)
        self.model.evaluate(layer1, layer2, batch_size=1)

        self.test_image(layer1)


if __name__ == "__main__":
    test_model = Model("", "", "")
    test_model.run()
