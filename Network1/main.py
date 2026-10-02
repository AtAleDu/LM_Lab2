import mnist_loader
import network


# Загружаем MNIST
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Преобразуем zip-объекты в списки
training_data = list(training_data)
validation_data = list(validation_data)
test_data = list(test_data)

# Создаем нейронную сеть:
# 784 входа -> 30 скрытых нейронов -> 10 выходов
net = network.Network([784, 30, 10])

# Обучаем сеть
net.SGD(
    training_data,
    30,     # количество эпох
    10,     # размер mini-batch
    3.0,    # скорость обучения
    test_data=test_data
)