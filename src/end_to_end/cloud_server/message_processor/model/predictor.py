import tensorflow as tf
from model_builder import ModelBuilder
import os
import cv2
import numpy as np
from typing import Callable, Any


class Predictor:

    def __init__(
            self,
            model_builder: ModelBuilder,
            dictionary_path: str,
            noise_dim: int):
        self.model_builder = model_builder
        self.noise_dim = noise_dim
        self.index_by_word = {}
        with open(str(dictionary_path)) as f:
            for i, word in enumerate(f.readlines()):
                stripped = word.strip()
                self.index_by_word[stripped] = i

    def restore_checkpoint(
            self,
            checkpoints_dir: str,
            checkpoint_prefix: str,
            checkpoint_number: int) -> None:
        self.generator = self.model_builder.build_generator()
        self.discriminator = self.model_builder.build_discriminator()
        self.generator_optimizer, self.discriminator_optimizer = \
            self.model_builder.get_optimizers()
        self.checkpoint = self.model_builder.get_checkpoint(
            checkpoints_dir, checkpoint_prefix, self.generator_optimizer,
            self.discriminator_optimizer, self.generator, self.discriminator)
        self.checkpoint.restore(
            os.path.join(
                checkpoints_dir,
                f'{checkpoint_prefix}-{checkpoint_number}'))

    def prediction_for_word(self, word: str) -> np.ndarray:
        noise = tf.random.normal([1, self.noise_dim])
        word_index = self.index_by_word[word]
        word_vec = tf.one_hot([word_index], len(self.index_by_word))
        generated = self.generator([noise, word_vec], training=False)
        return generated.numpy()[0]

    def prediction_for_sentence(self, sentence: str) -> np.ndarray:
        words = sentence.lower().split(' ')
        pred_sentence = []
        for word in words:
            try:
                pred = self.prediction_for_word(word)
            except KeyError:
                print(f'{word} is not in dictionary. No prediction made.')
                continue
            pred_sentence.append(pred)

        return np.array(pred_sentence)

    def _transform(self, x: float) -> float:
        if x < 0:
            return 0.0
        else:
            return x * 255.0

    def _get_element_transform(self) -> Any:
        return np.vectorize(self._transform)

    def video_for_sentence(self, sentence: str, video_path: str) -> None:
        pred_sentence = self.prediction_for_sentence(sentence)
        transformation = self._get_element_transform()
        pred_sentence = np.array(transformation(pred_sentence), dtype='uint8')
        avi_fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 2
        size = (pred_sentence.shape[2], pred_sentence.shape[3])
        writer = cv2.VideoWriter(
            video_path, avi_fourcc, fps, size, isColor=False)
        for word in pred_sentence:
            for frame in word:
                writer.write(frame)
        writer.release()
