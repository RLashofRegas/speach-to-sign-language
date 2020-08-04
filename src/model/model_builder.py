import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Optimizer
import os
from typing import Tuple


class ModelBuilder:

    def __init__(
            self, num_frames: int, frame_shape: Tuple[int, int],
            num_words: int, noise_dim: int, learning_rate: float):
        self.num_frames = num_frames
        self.frame_shape = frame_shape
        self.num_words = num_words
        self.noise_dim = noise_dim
        self.learning_rate = learning_rate

    def build_discriminator(self) -> tf.keras.Model:
        video = tf.keras.Input(
            (self.num_frames, self.frame_shape[0], self.frame_shape[1], 1))
        label = tf.keras.Input((self.num_words,))

        conv1 = layers.Conv2D(
            16, (5, 5), strides=(
                2, 2), padding='same', input_shape=[
                self.frame_shape[0], self.frame_shape[1]])(video)
        conv1a = layers.LeakyReLU()(conv1)
        conv1d = layers.Dropout(0.1)(conv1a)

        conv2 = layers.Conv2D(
            32, (5, 5),
            strides=(4, 4),
            padding='same')(conv1d)
        conv2a = layers.LeakyReLU()(conv2)
        conv2d = layers.Dropout(0.1)(conv2a)

        conv3 = layers.Conv2D(
            32, (5, 5),
            strides=(2, 2),
            padding='same')(conv2d)
        conv3a = layers.LeakyReLU()(conv3)
        conv3d = layers.Dropout(0.1)(conv3a)

        conv4 = layers.Conv2D(
            32, (5, 5),
            strides=(2, 2),
            padding='same')(conv3d)
        conv4a = layers.LeakyReLU()(conv4)
        conv4d = layers.Dropout(0.1)(conv4a)

        reshaped = layers.Reshape((self.num_frames, 2 * 2 * 32))(conv4d)
        lstm = layers.Bidirectional(layers.LSTM(128))(reshaped)
        concat = layers.Concatenate()([lstm, label])

        dense1 = layers.Dense(256)(concat)
        dense2 = layers.Dense(128)(dense1)

        output = layers.Dense(1)(dense2)

        model = tf.keras.Model(inputs=[video, label], outputs=output)
        return model

    def build_generator(self) -> tf.keras.Model:
        seed = tf.keras.Input((self.noise_dim,))
        label = tf.keras.Input((self.num_words,))

        concat = layers.Concatenate()([seed, label])

        dense1 = layers.Dense(self.num_frames * 8, use_bias=False)(concat)
        dense1n = layers.BatchNormalization()(dense1)
        dense1a = layers.LeakyReLU()(dense1n)

        reshaped1 = layers.Reshape((self.num_frames, 8))(dense1a)

        lstm = layers.Bidirectional(layers.LSTM(32, use_bias=False))(reshaped1)
        dense3 = layers.Dense(2 * 2 * 8, use_bias=False)(lstm)
        dense3n = layers.BatchNormalization()(dense3)
        dense3a = layers.LeakyReLU()(dense3n)

        reshaped2 = layers.Reshape((2, 2, 8))(dense3a)

        conv1 = layers.Conv2DTranspose(
            16, (5, 5), strides=(
                4, 4), use_bias=False, input_shape=[
                2, 2, 8])(reshaped2)
        conv1n = layers.BatchNormalization()(conv1)
        conv1a = layers.LeakyReLU()(conv1n)

        conv2 = layers.Conv2DTranspose(
            16, (5, 5), strides=(
                2, 2), use_bias=False, input_shape=[
                8, 8, 16])(conv1a)
        conv2n = layers.BatchNormalization()(conv2)
        conv2a = layers.LeakyReLU()(conv2n)

        flat = layers.Flatten()(conv2a)
        dense4 = layers.Dense(
            self.num_frames *
            self.frame_shape[0] *
            self.frame_shape[1],
            activation='tanh')(flat)

        output = layers.Reshape(
            (self.num_frames,
             self.frame_shape[0],
             self.frame_shape[1],
             1))(dense4)

        model = tf.keras.Model(inputs=[seed, label], outputs=output)
        return model

    def get_optimizers(self) -> Tuple[Optimizer, Optimizer]:
        discriminator_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        generator_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        return generator_optimizer, discriminator_optimizer

    def get_checkpoint(
            self,
            checkpoints_dir: str,
            checkpoint_prefix: str,
            generator_optimizer: Optimizer,
            discriminator_optimizer: Optimizer,
            generator: tf.keras.Model,
            discriminator: tf.keras.Model) -> tf.train.Checkpoint:
        prefix = os.path.join(checkpoints_dir, checkpoint_prefix)
        checkpoint = tf.train.Checkpoint(
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            generator=generator,
            discriminator=discriminator)
        return checkpoint
