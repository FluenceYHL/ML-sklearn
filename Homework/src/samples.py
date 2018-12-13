# package
import numpy
# self


def get_car():
    train_data = numpy.load('../dataSet/car.npy')
    numpy.random.shuffle(train_data)
    x = train_data[:, :6]
    y = train_data[:, 6]
    return x, y


def get_iris():
    train_data = numpy.load('../dataSet/iris.npy')
    numpy.random.shuffle(train_data)
    x = train_data[:, :4]
    y = train_data[:, 4:5].T.reshape(150)
    return x, y


def get_watermelon():
    train_data = numpy.load('../dataSet/watermelon3.0α.npy')
    numpy.random.shuffle(train_data)
    x = train_data[:, 0:2]
    y = train_data[:, 2]
    return x, y


if __name__ == '__main__':
    print('ok')
