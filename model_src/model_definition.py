import os
import random
from typing import Generator, Tuple

import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, InputLayer, UpSampling2D
from keras.models import Sequential
from keras.preprocessing.image import (ImageDataGenerator, img_to_array,
                                       load_img)
from numpy import ndarray
from skimage.color import lab2rgb, rgb2lab
from skimage.io import imsave


class Model:
    def __init__(
        self,
        model_name: str,
        test_input: str,
        test_output: str,
        train_input: str,
        batch_size: int,
        epochs: int,
    ) -> None:
        self.model = Sequential()
        self.model_name = model_name
        self.test_input = test_input
        self.test_output = test_output
        self.train_input = train_input
        self.tensorboard = TensorBoard(log_dir=f"model_src/models/logs")
        self.data_generator_instance = ImageDataGenerator(
            shear_range=0.2, zoom_range=0.2, rotation_range=20, horizontal_flip=True
        )
        self.batch_size = batch_size
        self.epochs = epochs

    def load_images(self) -> tuple[ndarray, ndarray]:
        images = []

        for image_name in os.listdir(self.train_input):
            image = img_to_array(load_img(self.train_input + image_name))
            images.append(image)

        images_arr = np.array(images, dtype=float)

        split = int(0.95 * len(images_arr))

        training_images = images_arr[:split]
        test_images = images_arr[split:]
        return training_images, test_images

    def images_generator(
        self, images: ndarray
    ) -> Generator[Tuple[ndarray, ndarray], None, None]:

        for batch in self.data_generator_instance.flow(
            images, batch_size=self.batch_size
        ):
            batch_lab_images = rgb2lab(batch)
            instensity_batch_layers = batch_lab_images[:, :, :, 0]
            color_batch_layers = batch_lab_images[:, :, :, 1:] / 128

            yield (
                instensity_batch_layers.reshape(instensity_batch_layers.shape + (1,)),
                color_batch_layers,
            )

    def create_nn(self) -> None:
        self.model.add(InputLayer(input_shape=(256, 256, 1)))
        self.model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
        self.model.add(Conv2D(64, (3, 3), activation="relu", padding="same", strides=2))
        self.model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
        self.model.add(
            Conv2D(128, (3, 3), activation="relu", padding="same", strides=2)
        )
        self.model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
        self.model.add(
            Conv2D(256, (3, 3), activation="relu", padding="same", strides=2)
        )
        self.model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))
        self.model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
        self.model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
        self.model.add(UpSampling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
        self.model.add(UpSampling2D((2, 2)))
        self.model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
        self.model.add(Conv2D(2, (3, 3), activation="tanh", padding="same"))
        self.model.add(UpSampling2D((2, 2)))
        self.model.compile(optimizer="rmsprop", loss="mse")

    def evaluate_model(self, images: ndarray) -> None:
        lab_images = rgb2lab((1.0 / 255) * images)
        intensity_layers = lab_images[:, :, :, 0]
        color_layers = lab_images[:, :, :, 1:]

        color_layers = color_layers / 128

        model_evaluation = self.model.evaluate(
            intensity_layers.reshape(intensity_layers.shape + (1,)),
            color_layers,
            batch_size=self.batch_size,
        )
        print(model_evaluation)

    def test_images(self) -> None:
        images = []

        for image_name in os.listdir(self.test_input):
            image = img_to_array(load_img(self.test_input + image_name))
            images.append(image)

        images_arr = np.array(images, dtype=float)
        lab_images = rgb2lab((1.0 / 255) * images_arr)
        input_layers = lab_images[:, :, :, 0]
        input_layers = input_layers.reshape(input_layers.shape + (1,))

        output_layers = self.model.predict(input_layers)
        output_layers = output_layers * 128

        for output_index in range(len(output_layers)):
            canvas = np.zeros((256, 256, 3))
            canvas[:, :, 0] = input_layers[output_index][:, :, 0]
            canvas[:, :, 1:] = output_layers[output_index]

            imsave(f"{self.test_output}image-{str(output_index)}.png", lab2rgb(canvas))

    def run(self) -> None:
        training_images, test_images = self.load_images()

        self.create_nn()
        self.model.fit(
            self.images_generator(training_images),
            callbacks=[self.tensorboard],
            epochs=self.epochs,
            steps_per_epoch=int((len(training_images) / self.batch_size) + 1),
        )

        self.evaluate_model(test_images)
        self.model.save(f"model_src/models/{self.model_name}")

        self.test_images()
