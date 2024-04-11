#Step 1: Load libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import tensorflow as tf
import keras
import keras.layers as layers
from keras.models import Sequential
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

from keras.layers import Dense
# from keras.utils import to_categorical  -not allowed as tensor flow is now included keras
from tensorflow.keras.utils import to_categorical

#Step 2: Load data # step 3Splitting data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
train_images = x_train
train_labels = y_train
print(train_images.shape)  # (60000, 28, 28)
print(train_labels.shape)  # (60000,)
test_images = x_test
test_labels = y_test

print(test_images.shape)  #

train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5
# Flatten the images.
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))
print(train_images.shape)  # (60000, 784)
print(test_images.shape)  # (10000, 784)

#Step 4: Build the model.
model = Sequential([Dense(64, activation='relu', input_shape=(784,)),
                    Dense(64, activation='relu'),
                    Dense(10, activation='softmax'),
                    ])
print(model.summary())

#step 5: compile model

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

#6:step train the model

hist=model.fit(
    train_images,
    to_categorical(train_labels),
    epochs=5,
    validation_split=0.2,
    batch_size=32)


#7.Testing the Model
model.evaluate(test_images,to_categorical(test_labels))

# Predict on the first 5 test images.
predictions = model.predict(test_images[:5])

# Print our model's predictions.
print(np.argmax(predictions, axis=1))  # [7, 2, 1, 0, 4]

# Check our predictions against the ground truths.
print(test_images[:1].shape)

first_image = test_images[:1]
first_image = np.array(first_image, dtype='float')
pixels = first_image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()


# visualizing losses and accuracy

plt.show(block=True)

# Plot accuracy
plt.figure()
plt.plot(hist.history["accuracy"],label="Train Accuracy",color="black")
plt.plot(hist.history["val_accuracy"],label="Validation Accuracy",color="red", linestyle="dashed")
plt.title("Model Accuracy", color="darkred")
plt.xlabel("Train Accuracy")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.show(block=True)
# plot loss
plt.figure()
plt.plot(hist.history["loss"],label="Train Loss",color="black")
plt.plot(hist.history["val_loss"],label="Validation Loss",color="red",linestyle="dashed")
plt.title("Model Loss",color="darkred")
plt.xlabel("Train Loss")
plt.ylabel("Validation Loss")
plt.legend()
plt.show(block=True)





