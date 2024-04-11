import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self):
        self.W = tf.Variable(np.random.randn(), name='weight')
        self.b = tf.Variable(np.random.randn(), name='bias')
        self.optimizer = tf.optimizers.Adam(learning_rate=0.01)

    def predict(self, x):
        return self.W * x + self.b

    def mean_squared_error(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    def fit(self, x, y, epochs=10000):
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                predictions = self.predict(x)
                loss = self.mean_squared_error(y, predictions)
            gradients = tape.gradient(loss, [self.W, self.b])
            self.optimizer.apply_gradients(zip(gradients, [self.W, self.b]))

            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch + 1}, Loss: {loss.numpy()}')

    def plot_regression_line(self, x, y):
        plt.scatter(x, y, label='Data Points')
        plt.plot(x, self.predict(x), color='red', label='Regression Line')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Simple Linear Regression')
        plt.legend()
        plt.show()


# Generate random data
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = 2 * x + 1 + np.random.randn(100) * 2

# Instantiate and fit the linear regression model
model = LinearRegression()
model.fit(x, y)

# Plot the regression line
model.plot_regression_line(x, y)
