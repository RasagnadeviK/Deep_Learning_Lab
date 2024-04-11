import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import my_library as ml
x_data = np.linspace(0, 2, 20)
y_data = 1.5 * x_data + np.random.randn(*x_data.shape) * 0.2 + 0.5

model = ml.LinearRegressionModel()
epochs = 100
learning_rate = 0.01
for epoch in range(epochs):
    loss = ml.train_step(model, x_data, y_data, learning_rate)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.numpy()}, m: {model.m.numpy()}, c: {model.c.numpy()}')
print("m: ", model.m.numpy())
print("c:", model.c.numpy())
ml.plot_LR_graph(model, x_data, y_data)