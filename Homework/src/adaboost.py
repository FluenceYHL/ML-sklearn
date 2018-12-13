# package
import time
import math
import numpy
# self


def sign_less_1(y, border):
    length = len(y)
    g = numpy.zeros((length))
    for i in range(length):
        g[i] = int(y[i] == -1) if(i <= border) else int(y[i] == 1)
    return g


def sign_more_1(y, border):
    length = len(y)
    g = numpy.zeros((length))
    for i in range(length):
        g[i] = int(y[i] == 1) if(i <= border) else int(y[i] == -1)
    return g


def easyClassfier(x, y, weights, sign):
    length = len(weights)   # x[border]
    error = 1e9
    index = 0
    for border in range(length - 1):
        g = sign(y, border)
        res = numpy.dot(g, weights)
        if(error > res):
            error = res
            index = border
            G = g
    alpha = 0.5 * math.log((1 - error) / error)
    for i in range(length):
        G[i] = y[i] if(G[i] == 0) else -y[i]

    weights = weights * numpy.exp(-alpha * (y * G))
    normal = sum(weights)
    weights /= normal

    return error, alpha, weights, index


class adaBoost():
    def fit(self, x, y, errorRate=0.1):
        length = len(x)
        self.weights = [1 / length] * length
        self.classfiers = []

        cnt = 0
        while(True):
            error_1, alpha_1, D_1, index_1 = easyClassfier(
                x, y, self.weights, sign_less_1)
            error_2, alpha_2, D_2, index_2 = easyClassfier(
                x, y, self.weights, sign_more_1)
            if(error_1 < error_2):
                self.weights = D_1
                self.classfiers.append([alpha_1, 0, index_1])
            else:
                self.weights = D_2
                self.classfiers.append([alpha_2, 1, index_2])
            error = self.score(x, y)
            if(error > 1 - errorRate):
                break

    def score(self, x, y):
        length = len(x)
        error = 0
        for i in range(length):
            judge = 0.0
            for it in self.classfiers:
                res = 0
                if(it[1] == 0):
                    res = 1 if(i <= it[2]) else -1
                else:
                    res = -1 if(i <= it[2]) else 1
                judge += it[0] * res
            judge = 1 if(judge > 0) else -1
            print(judge)
            error += int(judge == y[i])
        error /= length
        print(error)
        return error


if __name__ == '__main__':
    x = list(range(10))
    y = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]

    one = adaBoost()
    one.fit(x, y)
