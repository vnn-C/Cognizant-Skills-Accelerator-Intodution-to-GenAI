# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import tensorflow as tf
import os
import numpy as np
import cv2
import pandas as pd
from keras import layers
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from datetime import datetime

# -

#Variables
image_list = []
source_folder = "data/processed_gallery"
size = (128, 128)
shape_size = (128, 128, 3)
latent_dim = 100
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
#training for 700 epochs after would take over a day on my laptop
epochs = 600
batch_size = 16
save_check = 100

print("Fetching Data")
# +
#load images from folder
for file_name in os.listdir(source_folder):
    try:
        image = cv2.imread(os.path.join(source_folder, file_name))

        image = cv2.resize(image, size)

        image_list.append(image.astype(np.float32) / 255.0)


    except Exception as e:
        print(f"Error with loading image {os.path.join(source_folder, file_name)}: {e}")



data = np.array(image_list)
dataset = tf.data.Dataset.from_tensor_slices(data)
dataset = dataset.shuffle(buffer_size=1000).batch(batch_size, drop_remainder=True)

print("Building Models")
# +
#generator
def build_generator():
    model = Sequential([
        layers.Dense(8 * 8 * 512, activation="relu", input_dim=latent_dim),
        layers.Reshape((8, 8, 512)),

        layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding="same", activation="tanh"),
    ])
    return model

#discriminator
def build_discriminator():
    model = Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same", input_shape=shape_size),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.3),
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.3),
        layers.Conv2D(256, (5, 5), strides=(2, 2), padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.3),
        layers.Conv2D(512, (5, 5), strides=(2, 2), padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1, activation="sigmoid")
    ])

    return model

#saving images
def save_results(generator, epoch, test_data):
    predictions = generator(test_data, training=False)
    predictions = (predictions * 127.5 + 127.5).numpy().astype(np.uint8)

    dest = "data/results_gallery"

    for i, img in enumerate(predictions):
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        file_path = os.path.join(dest, f"epoch_{epoch}_image_{i}.jpg")

        cv2.imwrite(file_path, img_bgr)


# -

#loss functions
def discriminator_loss(real_out, fake_out):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_out), real_out)
    fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_out), fake_out)
    loss = real_loss + fake_loss
    return loss
def generator_loss(fake_out):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_out), fake_out)


generator = build_generator()
discriminator = build_discriminator()
optimizer_G = Adam(learning_rate=lr, beta_1=beta1, beta_2=beta2)
optimizer_D = Adam(learning_rate=lr, beta_1=beta1, beta_2=beta2)

# +
#Training loop

print("Starting Training")

for epoch in range(epochs):
    curr_date = datetime.now().strftime("%Y/%m/%d_%H:%M:%S")
    print(curr_date)
    print(f"Starting epoch {epoch}")
    count = 1
    for real in dataset.batch(batch_size):
        
        print(f"Epoch {epoch}, Batch {count}")
        count+=1
        real = tf.reshape(real, (-1, 128, 128, 3))

        real_resized = tf.image.resize(real, (128, 128))

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        noise = tf.convert_to_tensor(noise, dtype=tf.float32)
#train discriminator
        try:
            with tf.GradientTape() as disc_tape:


                fake = generator(noise, training=True)
                fake_resized = tf.image.resize(fake, (128, 128))

                real_result = discriminator(real_resized, training=True)
                fake_result = discriminator(fake_resized, training=True)

                disc_loss = discriminator_loss(real_result, fake_result)

            disc_gradient = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            optimizer_D.apply_gradients(zip(disc_gradient, discriminator.trainable_variables))
        except Exception as e:
            print(f"Error with discriminator training at epoch#{epoch}: {e}")
#train generator
        try:
            with tf.GradientTape() as gen_tape:
                new_fakes = generator(noise, training=True)
                new_fakes_resized = tf.image.resize(new_fakes, (128, 128))
                fake_result = discriminator(new_fakes_resized, training=True)

                gen_loss = generator_loss(fake_result)

            gen_gradient = gen_tape.gradient(gen_loss, generator.trainable_variables)
            optimizer_G.apply_gradients(zip(gen_gradient, generator.trainable_variables))
        except Exception as e:
            print(f"Error with generator training at epoch#{epoch}: {e}")

    if epoch % save_check == 0:
        curr_date = datetime.now().strftime("%Y/%m/%d_%H:%M:%S")
        print(curr_date)
        print(f"epoch {epoch}:\nDiscriinator loss: {disc_loss.numpy()}\nGenerator loss: {gen_loss.numpy()}")
        #save_results(generator, epoch, noise)
        

generator.save("models/gen_model.h5")
discriminator.save("models/disc_model.h5")
