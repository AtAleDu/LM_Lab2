"""
network.py
Модуль создания и обучения нейронной сети
для распознавания рукописных цифр.
"""

import random
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

class Network(object):
    def __init__(self, sizes):
        # Количество слоев
        self.num_layers = len(sizes)

        # Количество нейронов в каждом слое
        self.sizes = sizes

        # Случайные начальные смещения
        self.biases = [
            np.random.randn(y, 1)
            for y in sizes[1:]
        ]

        # Случайные начальные веса
        self.weights = [
            np.random.randn(y, x)
            for x, y in zip(sizes[:-1], sizes[1:])
        ]
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data):
        # Преобразуем тестовую выборку в список
        test_data = list(test_data)
        n_test = len(test_data)

        # Преобразуем обучающую выборку в список
        training_data = list(training_data)
        n = len(training_data)

        # Цикл по эпохам
        for j in range(epochs):

            # Перемешиваем обучающие данные
            random.shuffle(training_data)

            # Разбиваем данные на мини-выборки
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            # Обучаем сеть на каждой мини-выборке
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            # Показываем результат после каждой эпохи
            print(
                "Epoch {0}: {1} / {2}".format(
                    j,
                    self.evaluate(test_data),
                    n_test
                )
            )

    def update_mini_batch(self, mini_batch, eta):
        # Градиенты для смещений
        nabla_b = [
            np.zeros(b.shape)
            for b in self.biases
        ]

        # Градиенты для весов
        nabla_w = [
            np.zeros(w.shape)
            for w in self.weights
        ]

        # Проходим по всем примерам mini-batch
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            nabla_b = [
                nb + dnb
                for nb, dnb in zip(nabla_b, delta_nabla_b)
            ]

            nabla_w = [
                nw + dnw
                for nw, dnw in zip(nabla_w, delta_nabla_w)
            ]

        # Обновляем веса
        self.weights = [
            w - (eta / len(mini_batch)) * nw
            for w, nw in zip(self.weights, nabla_w)
        ]

        # Обновляем смещения
        self.biases = [
            b - (eta / len(mini_batch)) * nb
            for b, nb in zip(self.biases, nabla_b)
        ]

    def backprop(self, x, y):
        # Градиенты для смещений
        nabla_b = [
            np.zeros(b.shape)
            for b in self.biases
        ]

        # Градиенты для весов
        nabla_w = [
            np.zeros(w.shape)
            for w in self.weights
        ]

        # Прямое распространение
        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)

            activation = sigmoid(z)
            activations.append(activation)

        # Ошибка выходного слоя
        delta = (
            self.cost_derivative(activations[-1], y)
            * sigmoid_prime(zs[-1])
        )

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(
            delta,
            activations[-2].transpose()
        )

        # Обратное распространение ошибки
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)

            delta = (
                np.dot(
                    self.weights[-l + 1].transpose(),
                    delta
                )
                * sp
            )

            nabla_b[-l] = delta

            nabla_w[-l] = np.dot(
                delta,
                activations[-l - 1].transpose()
            )

        return (nabla_b, nabla_w)
    def cost_derivative(self, output_activations, y):
        return output_activations - y
    
    def evaluate(self, test_data):
        test_results = [
            (np.argmax(self.feedforward(x)), y)
            for (x, y) in test_data
        ]

        return sum(
            int(x == y)
            for (x, y) in test_results
        )

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Создаем сеть:
# 2 нейрона входного слоя
# 3 нейрона скрытого слоя
# 1 нейрон выходного слоя
net = Network([2, 3, 1])


print("Сеть net:")
print("Количество слоев:", net.num_layers)

for i in range(net.num_layers):
    print("Количество нейронов в слое", i, ":", net.sizes[i])

for i in range(net.num_layers - 1):
    print("W_", i + 1, ":")
    print(np.round(net.weights[i], 2))

    print("b_", i + 1, ":")
    print(np.round(net.biases[i], 2))

test_input = np.array([
    [1.0],
    [0.5]
])

print("Результат feedforward:")
print(net.feedforward(test_input))