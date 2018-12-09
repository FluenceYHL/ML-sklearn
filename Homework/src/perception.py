# package
import numpy
import warnings
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
# self

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    train_data = numpy.load('../dataSet/watermelon3.0α.npy')
    print(train_data)
    x = train_data[:, 0:2]
    y = train_data[:, 2]

    clf = Perceptron(fit_intercept=False, n_iter=30, shuffle=False)
    clf.fit(x, y)
    print('正确率  :  ' + str(clf.score(x, y)))

    positive = []
    nagative = []
    for it in train_data:
        if(it[2] == 1):
            positive.append([it[0], it[1]])
        else:
            nagative.append([it[0], it[1]])

    plt.scatter(positive[:1], positive[1:2],
                positive[2:3], c='red', marker='o')
    plt.scatter(nagative[:1], nagative[1:2], nagative[2:3], c='blue')
    line_x = numpy.arange(0, 1)
    line_y = line_x * (-clf.coef_[0][0] / clf.coef_[0][1]) - clf.intercept_
    plt.plot(line_x, line_y)
    plt.show()
