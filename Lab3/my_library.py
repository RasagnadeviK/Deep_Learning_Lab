import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn._loss import loss


def triangle_area(a, b, c):
    # Semi-perimeter
    s = (a + b + c) / 2.0

    # Heron's formula
    area = tf.sqrt(s * (s - a) * (s - b) * (s - c))
    return area
class LinearRegressionModel(tf.Module):
    def __init__(self):
        self.m = tf.Variable(np.random.normal(), dtype=tf.float32, name='M')
        self.c = tf.Variable(np.random.normal(), dtype=tf.float32, name='c')

    def __call__(self, x):
        return tf.add(tf.multiply(self.m, x), self.c)


def plot_LR_graph(model,x,y):
        y_pred=model(x)
        plt.scatter(x, y, color='black')
        plt.plot(x, y_pred, color='red')
        plt.xlim(0, 2)
        plt.ylim(0, 2)
        #plt.savefig('plot.png')
        plt.show(block=True)

def mean_squared_error(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true))

def train_step(model, inputs, targets, learning_rate=0.01):
    with tf.GradientTape() as tape:
            predictions = model(inputs)
            loss = mean_squared_error(targets, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer = tf.optimizers.SGD(learning_rate)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
