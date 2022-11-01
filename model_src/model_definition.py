import os
import random
from typing import Generator, Tuple

import numpy as np
import tensorflow as tf
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Input, InputLayer, UpSampling2D, Reshape, concatenate
from keras.layers.core import RepeatVector
from keras.models import Model
from keras.preprocessing.image import (ImageDataGenerator, img_to_array,
                                       load_img)
from numpy import ndarray
from skimage.color import lab2rgb, rgb2lab
from skimage.io import imsave


class ColorizationModel:
    def __init__(
        self,
        model_name: str,
        test_input: str,
        test_output: str,
        train_input: str,
        batch_size: int,
        epochs: int,
    ) -> None:
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

    def _load_images(self) -> tuple[ndarray, ndarray]:
        images = []

        for image_name in os.listdir(self.train_input):
            image = img_to_array(load_img(f"{self.train_input}{image_name}"))
            images.append(image)

        images_arr = np.array(images, dtype=float) * (1.0 / 255) # Multiplying by 1.0 / 255 reduces the color spectrum

        split = int(0.95 * len(images_arr))

        training_images = images_arr[:split]
        test_images = images_arr[split:]
        return training_images, test_images

    def _load_resnet_weights(self) -> InceptionResNetV2:
        inception = InceptionResNetV2(weights="imagenet", include_top=True)
        inception.graph = tf.get_default_graph()
        return inception

    def _images_generator(
        self, images: ndarray
    ) -> Generator[Tuple[list[ndarray, ndarray], ndarray], None, None]:
        inception = self._load_resnet_weights()

        for batch in self.data_generator_instance.flow(
            images, batch_size=self.batch_size
        ):
            sample_grayscaled_rgb = gray2rgb(rgb2gray(batch))
            embedding = self._create_resnet_embedding(sample_grayscaled_rgb, inception)
            batch_lab_images = rgb2lab(batch)
            intensity_batch_layers = batch_lab_images[:, :, :, 0]
            color_batch_layers = batch_lab_images[:, :, :, 1:] / 128

            yield (
                [
                    intensity_batch_layers.reshape(intensity_batch_layers.shape + (1,)),
                    embedding
                ],
                color_batch_layers,
            )

    def _create_nn(self) -> Model:
        embed_input = Input(shape=(1000,))

        # Encoder Layers of the Neural Network
        encoder_input = Input(shape=(256, 256, 1,))
        encoder_output = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(encoder_input)
        encoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
        encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
        encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
        encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)

        # Fusion Layers of the Neural Network
        fusion_output = RepeatVector(32 * 32)(embed_input) 
        fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
        fusion_output = concatenate([encoder_output, fusion_output], axis=3) 
        fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output) 

        # Decoder Layers of the Neural Network
        decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(fusion_output)
        decoder_output = UpSampling2D((2, 2))(decoder_output)
        decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
        decoder_output = UpSampling2D((2, 2))(decoder_output)
        decoder_output = Conv2D(32, (3,3), activation='relu', padding='same')(decoder_output)
        decoder_output = Conv2D(16, (3,3), activation='relu', padding='same')(decoder_output)
        decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
        decoder_output = UpSampling2D((2, 2))(decoder_output)

        model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)

        return model
    
    def _create_resnet_embedding(self, sample_grayscaled_rgb, inception) -> ndarray:
        sample_grayscaled_rgb_resized = []

        for img in sample_grayscaled_rgb:
            img = resize(img, (299, 299, 3), mode="constant")
            sample_grayscaled_rgb_resized.append(img)
        
        sample_grayscaled_rgb_resized = preprocess_input(np.array(sample_grayscaled_rgb_resized))

        with inception.graph.as_default():
            embedding = inception.predict(sample_grayscaled_rgb_resized)
        
        return embedding

    def _evaluate_model(self, model: Model, images: ndarray) -> None:
        inception = self._load_resnet_weights()
        grayed_rgb_images = gray2rgb(rgb2gray(images))
        lab_images = rgb2lab(images)
        intensity_layers = lab_images[:, :, :, 0]
        color_layers = lab_images[:, :, :, 1:] / 128

        model_evaluation = model.evaluate(
            [
                intensity_layers.reshape(intensity_layers.shape + (1,)),
                self._create_resnet_embedding(grayed_rgb_images, inception)
            ],
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
        training_images, test_images = self._load_images()

        model = self._create_nn()
        model.compile(optimizer="rmsprop", loss="mse")
        model.fit_generator(
            self._images_generator(training_images),
            callbacks=[self.tensorboard],
            epochs=self.epochs,
            steps_per_epoch=int((len(training_images) / self.batch_size) + 1),
        )

        self._evaluate_model(model, test_images)
        model.save(f"model_src/models/{self.model_name}")

        # self.test_images()
