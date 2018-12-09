# package
import numpy
import warnings
from sklearn import svm
import matplotlib.pyplot as plt
# self

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    train_data = numpy.load('../dataSet/watermelon3.0α.npy')
    x = train_data[:, 0:2]
    y = train_data[:, 2]

    classfier = svm.SVC(C=300)
    classfier.fit(x, y)
    print(classfier.score(x, y))
    print(classfier.support_vectors_)

    classfier2 = svm.SVC(kernel='rbf', C=5000)
    classfier2.fit(x, y)
    print(classfier2.score(x, y))
    print(classfier2.support_vectors_)

    classfier3 = svm.SVC(kernel='linear', C=5000)
    classfier3.fit(x, y)
    print(classfier3.score(x, y))
    print(classfier3.support_vectors_)

    classfier4 = svm.SVC(kernel='poly', C=5000)
    classfier4.fit(x, y)
    print(classfier4.score(x, y))
    print(classfier4.support_vectors_)
