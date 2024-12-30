#!/usr/bin/env python
# coding: utf-8

# Importing Libraries

# In[14]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[12]:


#load the allpix.csv data
all_pix = pd.read_csv('allpix.csv', header=None)

#convert to numpy array
all_pix = all_pix.values

#normalize pixel values between [0, 1]
all_pix = (all_pix + 1) / 2

#reshape the data to (samples, 15, 15, 1) for CNN input
all_pix = all_pix.reshape(-1, 15, 15, 1)

#split the data into training and testing sets
train_pix, test_pix = train_test_split(all_pix, test_size=0.2, random_state=42)

#build the autoencoder model
def build_autoencoder():
    input_img = layers.Input(shape=(15, 15, 1))

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)

    # Adjust the output size to 15x15 using valid padding or Cropping2D
    x = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    decoded = layers.Cropping2D(cropping=((0, 1), (0, 1)))(x)  # Crop down to (15, 15, 1)

    autoencoder = models.Model(input_img, decoded)
    return autoencoder

#build and compile the autoencoder
autoencoder = build_autoencoder()
autoencoder.compile(optimizer='adam', loss='mse')

#train the autoencoder
history = autoencoder.fit(train_pix, train_pix, epochs=50, batch_size=32, validation_data=(test_pix, test_pix))

#plot loss curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#evaluate the model
loss = autoencoder.evaluate(test_pix, test_pix)
print(f"Test Loss: {loss:.4f}")

#make predictions on test data
decoded_imgs = autoencoder.predict(test_pix)


# In[13]:


#display original and reconstructed images
n = 5  #number of images to display
plt.figure(figsize=(10, 4))
for i in range(n):
    #display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(test_pix[i].reshape(15, 15), cmap='gray')
    plt.title("Original")
    plt.axis("off")

    #display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(15, 15), cmap='gray')
    plt.title("Reconstructed")
    plt.axis("off")
plt.show()

