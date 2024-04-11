import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import regularizers

class CNNModel(tf.keras.Model):
    def __init__(self, input_shape=(32, 32, 3), num_classes=10):
        super(CNNModel, self).__init__()
        self.conv1 = Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)
        self.pool1 = MaxPooling2D((2, 2))
        self.conv2 = Conv2D(64, (3, 3), activation='relu')
        self.pool2 = MaxPooling2D((2, 2))
        self.conv3 = Conv2D(64, (3, 3), activation='relu')
        self.pool3 = MaxPooling2D((2, 2))
        self.flatten = Flatten()
        self.dense1 = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))
        self.dropout = Dropout(0.5)
        self.dense2 = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)

# Instantiate the model
model = CNNModel()

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.build((None, 32, 32, 3))  # Manually build the model to specify input shape
print(model.summary())
