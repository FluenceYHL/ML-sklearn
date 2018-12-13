# package
import time
import math
import numpy
# self


class binaryJudge():
    def find(self, x, y, weights, l, r, normal=1e-5):
        length = len(weights)   # x[border]
        error = 1e9
        index = 0
        for border in range(int(max(x))):   # 以　x 取值范围设定边界
            g = numpy.zeros((length))  # 如果以　border 为界，g 用来标记分类错误的点
            for i in range(length):
                g[i] = int(y[i] == l) if(x[i] <= border) else int(
                    y[i] == r)  # g 中分类错误的点被标成　1
            res = numpy.dot(g, weights)
            if(error > res):
                error = res
                index = border  # 从中取错误最小的分界, 记录分界点, 以及错误最小对应的　　错误点
                G = g
        # 防止　0 化
        error += normal
        # 在这里调整本分类器的投票权重
        alpha = 0.5 * math.log((1 - error) / error)
        for i in range(length):
            G[i] = y[i] if(G[i] == 0) else -y[i]  # 恢复最小错误对应的判断　

        # y * G 不同号，这一项就更大, 错误的点获取更多的关注
        weights = weights * numpy.exp(-alpha * (y * G))
        normal = sum(weights)
        weights /= normal

        return error, alpha, weights, index

    def fit(self, x, y, weights):
        # <= 1, > 1 和　<= -1 > 1 找最好的
        lhs, alpha1, weights1, index1 = self.find(x, y, weights, -1, 1)
        rhs, alpha2, weights2, index2 = self.find(x, y, weights, 1, -1)
        if(lhs < rhs):
            self.less = True
            self.border = index1
            return alpha1, weights1
        else:
            self.less = False
            self.border = index2
            return alpha2, weights2

    def predict(self, x):
        if(self.less == True):
            return 1 if(x <= self.border) else -1
        else:
            return -1 if(x <= self.border) else 1


class adaBoost():
    def fit(self, x, y, accuracy=0.9):
        length = len(x)
        self.classfiers = []
        weights = [1 / length] * length

        while(True):
            one = binaryJudge()
            alpha, weights = one.fit(x, y, weights)
            self.classfiers.append([alpha, one])
            print(one.border)
            print(alpha)
            print(weights)
            if(self.score(x, y) >= accuracy):
                break

    def score(self, x, y):
        accuracy = 0
        length = len(x)
        for i in range(length):
            judge = 0.0
            for it in self.classfiers:
                judge += it[0] * it[1].predict(x[i])  # alpha * 分类器 it[1]
            judge = 1 if(judge > 0) else -1
            accuracy += int(judge == y[i])
        accuracy /= length
        print('正确率 ' + str(accuracy))
        return accuracy


if __name__ == '__main__':
    # x = list(range(10))
    # y = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]

    # one = adaBoost()
    # one.fit(x, y, accuracy=0.9)

    x = list(range(10))
    y = [1, 1, 1, 1, -1, -1, -1, 1, 1, 1]

    one = adaBoost()
    one.fit(x, y, accuracy=0.9)
