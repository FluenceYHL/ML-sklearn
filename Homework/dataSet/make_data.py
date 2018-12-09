# package
import numpy


if __name__ == '__main__':
    # obj = open('./iris.txt')
    # lines = obj.readlines()
    # iris_data = []
    # for it in lines:
    #     it = it.replace('\n', '')
    #     feature = it.split(',')
    #     iris_data.append(feature)
    # iris_data = numpy.array(iris_data)
    # # print(iris_data)
    # numpy.save('./iris.npy', iris_data)
    # obj.close()

    iris_data = numpy.load('./iris.npy')
    print(iris_data)
