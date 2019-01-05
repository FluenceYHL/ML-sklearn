'''
*********************************************************************************
  *Copyright(C),Your Company
  *FileName:         6_maxEnt.py
  *Author:           刘畅
  *Version:          1.0
  *Date:             2019.1.3
  *Description:      机器学习实验六  最大熵  调用了 nltk 的 classfiers 中的 maxent 包
  *Compile/Explain:  python3 6_maxEnt.py 
  *Function List:
  *Item List:
  *History:
     1.Date:         2019.1.3
       Author:       刘畅
       Modification: 完成了基本功能
       Problem:      
**********************************************************************************
'''
# package
import os
import nltk
import numpy
import warnings
import datetime
import matplotlib.pyplot as plt
from memory_profiler import profile
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
# self
import samples


@profile(precision=4, stream=open('./logs/6_maxEnt.log', 'w+'))
def solve(x, y, feature_names, _algorithm, _trace=0, _max_iter=100, pca=False, d=2, dtype='data'):
    if(pca is True):
        x = PCA(n_components=d).fit_transform(x)
    train_data, test_data = [], []
    start = datetime.datetime.now()
    # 生成特征函数必要的计数信息, 合成训练样本 (feature, label)
    # feature 代表 {a: 1, b: 2, c: 3} 之类的特征信息
    for one, label in zip(x, y):
        feature = {l: r for l, r in zip(feature_names, one)}
        train_data.append((feature, label))
        test_data.append(feature)
    # 训练采用 ISS 迭代
    solver = nltk.classify.MaxentClassifier.train(
        train_data, algorithm=_algorithm, trace=_trace, max_iter=_max_iter)
    print('耗时 ', datetime.datetime.now() - start, ' s')
    # 开始测试拟合程度
    res = (solver.classify(it) for it in test_data)
    ans = sum(l == r for l, r in zip(res, y))
    print('拟合程度  :  ', ans / len(y))

    if(pca is True):
        names = list(set(y))
        trans = {names[i]: i for i in range(len(names))}
        x1_min, x1_max = numpy.min(x[:, 0]), numpy.max(x[:, 0])
        x2_min, x2_max = numpy.min(x[:, 1]) - 1, numpy.max(x[:, 1]) + 1
        xx1, xx2 = numpy.meshgrid(numpy.arange(
            x1_min, x1_max, 0.1), numpy.arange(x2_min, x2_max, 0.1))
        data = numpy.c_[xx1.ravel(), xx2.ravel()]
        inpt = [{l: r for l, r in zip(feature_names, it)} for it in data]
        Z = [solver.classify(it) for it in inpt]
        # 现在开始编码
        Z = numpy.array([trans[it] for it in Z])
        Z = Z.reshape(xx1.shape)
        print(Z)
        plt.contour(xx1, xx2, Z, cmap=ListedColormap(['red']))
        print(Z)
        print(x.shape)
        for it in names:
            kind = numpy.array([l for l, r in zip(x, y) if(r == it)])
            print(kind.shape)
            plt.scatter(kind[:, 0], kind[:, 1])
        plt.title('鸢尾花主要特征分布图')
        plt.xlabel('新特征 X')
        plt.ylabel('新特征 Y')
        plt.legend(names, loc='lower right')
        plt.show()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    print('拟合鸢尾花数据集  :  \n')
    x, y = samples.get_iris()
    feature_names = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']
    solve(x, y, feature_names, _algorithm='IIS',
          _trace=0, _max_iter=20, dtype='data')
    solve(x, y, _algorithm='IIS', feature_names=['花萼长度', '花萼宽度'],
          _trace=0, _max_iter=20, pca=True, dtype='data')

    # print('\n\n拟合 car 数据集  :  \n')
    # x, y = samples.get_car()
    # feature_names = ['最高时速', '排量', '经济指数', '稳定指数', '容量', '价格']
    # print('编码降维之前  :  \n')
    # solve(x, y, feature_names, _algorithm='IIS',
    #       _trace=0, _max_iter=120, pca=False, dtype='string')
    # print('\n编码降维之后  :  \n')
    # solve(x, y, feature_names, _algorithm='IIS',
    #       _trace=0, _max_iter=120, pca=True, dtype='string')

'''
拟合鸢尾花数据集  :  

耗时  0:00:01.839704  s
拟合程度  :   1.0


拟合 car 数据集  :  

编码降维之前  :  

耗时  0:03:27.547675  s
拟合程度  :   0.8877314814814815

编码降维之后  :  

耗时  0:01:50.952534  s
拟合程度  :   0.9861111111111112

YHL make it !
'''
