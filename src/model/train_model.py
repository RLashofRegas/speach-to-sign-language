import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from pathlib import Path
import os


buffer_size = 30000
batch_size = 64
dataset_root = Path('dataset')
dictionary_path = Path('dictionary.txt')
num_frames = 253
frame_shape = (500, 500)
input_shape = [batch_size, num_frames, frame_shape[0], frame_shape[1]]
epochs = 500
noise_dim = 100
num_examples_to_generate = 32

file_batches = tf.data.Dataset.list_files(
    str(dataset_root / '*/*/*.avi')).shuffle(buffer_size).batch(batch_size)

index_by_word = {}
with open(str(dictionary_path)) as f:
    for i, word in enumerate(f.readlines()):
        stripped = word.strip()
        index_by_word[stripped] = i

num_words = len(index_by_word)


def make_discriminator_model():
    video = tf.keras.Input((num_frames, frame_shape[0], frame_shape[1], 1))
    label = tf.keras.Input((num_words,))

    conv1 = layers.Conv2D(
        64, (5, 5),
        strides=(2, 2),
        padding='same', input_shape=input_shape[2:])(video)
    conv1a = layers.LeakyReLU()(conv1)
    conv1d = layers.Dropout(0.1)(conv1a)

    conv2 = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(conv1d)
    conv2a = layers.LeakyReLU()(conv2)
    conv2d = layers.Dropout(0.1)(conv2a)

    conv3 = layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same')(conv2d)
    conv3a = layers.LeakyReLU()(conv3)
    conv3d = layers.Dropout(0.1)(conv3a)

    conv4 = layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same')(conv3d)
    conv4a = layers.LeakyReLU()(conv4)
    conv4d = layers.Dropout(0.1)(conv4a)

    conv5 = layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same')(conv4d)
    conv5a = layers.LeakyReLU()(conv5)
    conv5d = layers.Dropout(0.1)(conv5a)

    reshaped = layers.Reshape((num_frames, 16*16*256))(conv3d)
    lstm = layers.Bidirectional(layers.LSTM(256))(reshaped)
    concat = layers.Concatenate()([lstm, label])

    dense1 = layers.Dense(256)(concat)
    dense2 = layers.Dense(128)(dense1)

    output = layers.Dense(1)(dense2)
    
    model = tf.keras.Model(inputs=[video, label], outputs=output)
    return model


def make_generator_model():
    seed = tf.keras.Input((noise_dim,))
    label = tf.keras.Input((num_words,))

    concat = layers.Concatenate()([seed, label])
    
    dense1 = layers.Dense(256, use_bias=False)(concat)
    dense1n = layers.BatchNormalization()(dense1)
    dense1a = layers.LeakyReLU()(dense1n)

    dense2 = layers.Dense(num_frames*256, use_bias=False)(dense1)
    dense2n = layers.BatchNormalization()(dense2)
    dense2a = layers.LeakyReLU()(dense2n)

    lstm = layers.Bidirectional(layers.LSTM(256, use_bias=False))(dense2)
    dense3 = layers.Dense(num_frames*16*16*256, use_bias=False)
    dense3n = layers.BatchNormalization()(dense3)
    dense3a = layers.LeakyReLU()(dense3n)

    reshaped = layers.Reshape((num_frames, 16, 16, 256))(dense3a)

    conv1 = layers.Conv2DTranspose(256, (5,5), strides=(2,2), use_bias=False)(reshaped)
    conv1n = layers.BatchNormalization()(conv1)
    conv1a = layers.LeakyReLU()(conv1n)

    conv2 = layers.Conv2DTranspose(256, (5,5), strides=(2,2), use_bias=False)(conv1a)
    conv2n = layers.BatchNormalization()(conv2)
    conv2a = layers.LeakyReLU()(conv2n)

    conv3 = layers.Conv2DTranspose(256, (5,5), strides=(2,2), use_bias=False)(conv2a)
    conv3n = layers.BatchNormalization()(conv3)
    conv3a = layers.LeakyReLU()(conv3n)

    conv4 = layers.Conv2DTranspose(128, (5,5), strides=(2,2), use_bias=False)(conv3a)
    conv4n = layers.BatchNormalization()(conv4)
    conv4a = layers.LeakyReLU()(conv4n)

    dense4 = layers.Dense(num_frames*frame_shape[0]*frame_shape[1], activation='tanh')

    output = layers.Reshape((num_frames, frame_shape[0], frame_shape[1], 1))

    model = tf.keras.Model(inputs=[seed, label], outputs=output)
    return model

generator = make_generator_model()
discriminator = make_discriminator_model()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


seed = tf.random.normal([num_examples_to_generate, noise_dim])

rand_gen = tf.random.Generator.from_seed(1234)
rand = rand_gen.uniform([num_examples_to_generate], minval=0, maxval=num_words-1, dtype=tf.dtypes.int32)
test_labels = tf.one_hot(rand, num_words)


@tf.function
def train_step(images, labels):
    noise = tf.random.normal([batch_size, noise_dim])
    rand = rand_gen.uniform([batch_size], minval=0, maxval=num_words-1, dtype=tf.dtypes.int32)
    gen_labels = tf.one_hot(rand, num_words)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator([noise, gen_labels], training=True)

      real_output = discriminator([images, labels], training=True)
      fake_output = discriminator([generated_images, gen_labels], training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def get_video_data(file_batch):
    train_data = []
    train_labels = []
    for file_path in file_batch:
        train_file = Path(file_path)
        label = train_file.parts[-2]
        label_index = index_by_word[label]

        cap = cv2.VideoCapture(str(train_file))
        video_data = []
        while (True):
            ret, frame = cap.read()
            if (frame is None):
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            normalized_frame = gray_frame / 255.0
            video_data.append(normalized_frame)
        cap.release()

        train_data.append(video_data)
        train_labels.append(label_index)

    train_data = np.array(train_data).reshape((batch_size, num_frames, frame_shape[0], frame_shape[1], 1))
    train_labels = np.array(train_labels)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    return train_dataset


def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for file_batch in file_batches:
      video_batch = get_video_data(file_batch)
      train_labels_batch = tf.one_hot(video_batch[1], num_words)
      train_step(video_batch[0], train_labels_batch)


    # Save the model every 15 epochs
    if (epoch + 1) % 10 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
