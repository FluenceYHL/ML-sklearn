# package
import numpy
from sklearn import tree
# self

if __name__ == '__main__':
    train_data = numpy.load('../dataSet/watermelon3.0α.npy')
    x = train_data[:, 0:2]
    y = train_data[:, 2]

    classfier = tree.DecisionTreeClassifier()
    classfier.fit(x, y)
    print(classfier.score(x, y))
