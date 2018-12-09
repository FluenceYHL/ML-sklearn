# package
import cv2
import math
import time
import numpy
import warnings
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1.0 + numpy.exp(-x))


def dSigmoid(x):
    return x * (1.0 - x)


class BPNN():
    def __init__(self, inputSize=784, hideSize=100, outputSize=10, lrt=0.02):
        self.inputSize = inputSize
        self.hideSize = hideSize
        self.outputSize = outputSize
        self.lrt = lrt
        self.whid = numpy.random.uniform(-0.5,
                                         0.5, (self.inputSize, self.hideSize))
        self.wout = numpy.random.uniform(-0.5,
                                         0.5, (self.hideSize, self.outputSize))
        self.hid_out = numpy.zeros(self.inputSize)
        self.out_out = numpy.zeros(self.outputSize)

    def forward(self, inpt):
        self.hid_out = sigmoid(numpy.dot(inpt, self.whid))
        self.out_out = sigmoid(numpy.dot(self.hid_out, self.wout))
        return self.out_out

    def backward(self, inpt, target):
        out_daoshu = (target - self.out_out) * dSigmoid(self.out_out)
        self.wout += self.lrt * \
            self.hid_out.reshape(self.hideSize, 1) * out_daoshu
        hid_daoshu = dSigmoid(self.hid_out) * numpy.dot(self.wout, out_daoshu)
        self.whid += self.lrt * inpt.reshape(self.inputSize, 1) * hid_daoshu

    def recognize(self, inpt):
        return numpy.argmax(self.forward(inpt))


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('ignore')
    train_data = numpy.load('../dataSet/watermelon3.0α.npy')
    x = train_data[:, 0:2]
    y = train_data[:, 2]

    bpnn = BPNN(2, 20, 2, 3)
    for i in range(40):
        count = 0
        for it in x:
            target = numpy.zeros(2)
            target[int(y[count])] = 1
            bpnn.forward(it)
            bpnn.backward(it, target)
            count = count + 1

        count = 0
        correct = 0
        for it in x:
            if(bpnn.recognize(it) == int(y[count])):
                correct = correct + 1
            count = count + 1
        print('正确率  :  ' + str(correct / count))
