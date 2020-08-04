import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from pathlib import Path
import os
import time
import argparse
from model_builder import ModelBuilder

arg_parser = argparse.ArgumentParser(
    description='Run the preprocessing pipeline.'
)
arg_parser.add_argument(
    '-d', '--dictionary_path', type=str, default='dictionary.txt',
    help='Path to where dictionary will be created.')
arg_parser.add_argument(
    '-s', '--dataset_path', type=str, default='dataset',
    help='Path of where to look for the dataset.')
arg_parser.add_argument(
    '-c', '--checkpoint', type=str, default=None,
    help='Checkpoint to start from.'
)
args = arg_parser.parse_args()

buffer_size = 30000
batch_size = 16
dataset_root = Path(args.dataset_path)
dictionary_path = Path(args.dictionary_path)
num_frames_initial = 253
frame_sampling_rate = 15
num_frames = int(num_frames_initial / frame_sampling_rate) + 1
frame_shape = (64, 64)
input_shape = [batch_size, num_frames, frame_shape[0], frame_shape[1]]
num_epochs = 500
noise_dim = 20
num_examples_to_generate = 8
learning_rate = 1e-4

file_batches = tf.data.Dataset.list_files(
    str(dataset_root / '*/*/*.avi')).shuffle(buffer_size).batch(batch_size)

words_in_dir = set()
for word_dir in dataset_root.glob('*/*'):
    words_in_dir.add(word_dir.parts[-1])

index_by_word = {}
with open(str(dictionary_path)) as f:
    for i, word in enumerate(f.readlines()):
        stripped = word.strip()
        if (stripped in words_in_dir):
            index_by_word[stripped] = i

num_words = len(index_by_word)

model_builder = ModelBuilder(
    num_frames,
    frame_shape,
    num_words,
    noise_dim,
    learning_rate)

generator = model_builder.build_generator()
discriminator = model_builder.build_discriminator()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer, discriminator_optimizer = model_builder.get_optimizers()


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = 'ckpt'
checkpoint = model_builder.get_checkpoint(
    checkpoint_dir, checkpoint_prefix, generator_optimizer,
    discriminator_optimizer, generator, discriminator)

if (args.checkpoint is not None):
    checkpoint.restore(args.checkpoint)


seed = tf.random.normal([num_examples_to_generate, noise_dim])

rand_gen = tf.random.Generator.from_seed(1234)
rand = rand_gen.uniform(
    [num_examples_to_generate],
    minval=0,
    maxval=num_words - 1,
    dtype=tf.dtypes.int32)
test_labels = tf.one_hot(rand, num_words)


@tf.function
def train_step(images, labels):
    noise = tf.random.normal([batch_size, noise_dim])
    rand = rand_gen.uniform(
        [batch_size],
        minval=0,
        maxval=num_words - 1,
        dtype=tf.dtypes.int32)
    gen_labels = tf.one_hot(rand, num_words)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator([noise, gen_labels], training=True)

        real_output = discriminator([images, labels], training=True)
        fake_output = discriminator(
            [generated_images, gen_labels],
            training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables))


def get_video_data(file_batch):
    train_data = np.empty(
        (batch_size,
         num_frames,
         frame_shape[0],
         frame_shape[1],
         1))
    train_labels = np.zeros((batch_size, num_words))
    for file_index, file_path in enumerate(file_batch):
        train_file = Path(file_path.numpy().decode('utf-8'))
        label = train_file.parts[-2]
        label_index = index_by_word[label]

        cap = cv2.VideoCapture(str(train_file))
        video_data = np.zeros((num_frames, frame_shape[0], frame_shape[1], 1))
        frame_num = -1
        while (True):
            ret, frame = cap.read()
            frame_num += 1
            frame_mod = frame_num % frame_sampling_rate
            if (frame_mod != 0):
                continue
            if (frame is None):
                break

            frame_index = int(frame_num / frame_sampling_rate)

            resized = cv2.resize(frame, frame_shape)
            gray_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            normalized_frame = (
                gray_frame /
                255.0).reshape(
                (gray_frame.shape[0],
                 gray_frame.shape[1],
                 1))
            video_data[frame_index] = normalized_frame

        cap.release()

        train_data[file_index] = video_data
        train_labels[file_index, label_index] = 1

    return train_data, train_labels


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for file_batch in file_batches:
            train_data, train_labels = get_video_data(file_batch)
            train_step(train_data, train_labels)

        # Save the model every 15 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(
            epoch + 1, time.time() - start))


train(file_batches, num_epochs)
