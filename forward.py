'''
*********************************************************************************
  *Copyright(C),Your Company
  *FileName:     	 hmm.py
  *Author:       	 刘畅
  *Version:      	 1.0
  *Date:         	 2018.12.28
  *Description:  	 模拟实现隐马模型的前向后向算法，以及前向后向，维特比算法
  *Compile:      	 python3 hmm.py
  *Function List:  
  *Item List:
					 1. dispersed_hmm 简易离散型　hmm 模板
  *History:  
     1.Date:         2018.12.28
       Author:       刘畅
       Modification: 完成了基本功能
       Problem:     
**********************************************************************************
'''
# package
import numpy
# self


class dispersed_hmm():
    def fit(self, A, B, P):
        self.A, self.B, self.P = A, B, P
        self.n_components, self.n_result = A.shape[0], B.shape[1]

    # 前向算法
    def forward(self, observation):
        self.alpha = []
        self.alpha.append(P * self.B[:, observation[0]])
        for t in range(1, len(observation)):
            # self.alpha[t - 1],前一时刻每个状态的概率
            # self.A[:, i]) 在这一时刻转移到状态　i 的概率( 汇集　)
            # self.B[i][observation[t] 这一时刻状态 i, 产生结果为　observation[t] 的概率
            self.alpha.append([numpy.dot(self.alpha[t - 1], self.A[:, i]) * self.B[i]
                               [observation[t]] for i in range(self.n_components)])
        return sum(self.alpha[len(observation) - 1])

    # 后向算法
    def backward(self, observation):
        L = len(observation)
        self.belta = numpy.zeros((L, self.n_components))
        self.belta[L - 1] = 1
        for t in range(L - 2, -1, -1):
            # self.A[i] 当前状态为　i, 转移到下一时刻所有状态的概率序列
            # self.B[:, observation[t + 1]], 生成结果为　observation[t + 1] 的概率
            # self.belta[t + 1] 下一时刻的　self.belta
            self.belta[t] = [numpy.dot(self.A[i], self.B[:, observation[t + 1]]
                                       * self.belta[t + 1]) for i in range(self.n_components)]
        return numpy.dot(self.P, self.B[:, observation[0]] * self.belta[0])

    # 选择前向或者后向计算某一个序列的概率
    def score(self, observation, algorithm='forward'):
        return self.forward(observation) if(algorithm is 'forward') else self.backward(observation)

    # 利用前向后向的性质求出序列位置为　t 的观察结果为　state 的概率
    def forward_back(self, observation, t, state):
        self.forward(observation)
        self.backward(observation)
        S = self.alpha[t] * self.belta[t]
        return S[state] / sum(S)

    # 利用　viterbi 算法预测(解码) observation, 得到标注序列
    def decode(self, observation, algorithm='viterbi'):
        if(algorithm is 'viterbi'):
            L = len(observation)
            likely = []
            path = [[0] * self.n_components]
            likely.append(self.P * self.B[:, observation[0]])
            for t in range(1, L):
                probabity = [likely[t - 1] * self.A[:, i]
                             for i in range(self.n_components)]
                path.append([numpy.argmax(it) for it in probabity])
                likely.append(
                    [max(l) * r for l, r in zip(probabity, self.B[:, observation[t]])])
            res = [0] * L
            res[L - 1] = numpy.argmax(likely[L - 1])
            for i in range(L - 2, -1, -1):
                res[i] = path[i + 1][res[i + 1]]
            return res


if __name__ == '__main__':
    A = numpy.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    B = numpy.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
    P = [0.2, 0.4, 0.4]
    one = dispersed_hmm()
    one.fit(A, B, P)
    print('前向算法　　:  ', one.score([0, 1, 0], algorithm='forward'))
    print('后向算法　　:  ', one.score([0, 1, 0], algorithm='backward'))
    print('预测序列(从　0 数起)　　:  ', one.decode([0, 1, 0], algorithm='viterbi'))

    A = numpy.array([[0.5, 0.1, 0.4], [0.3, 0.5, 0.2], [0.2, 0.2, 0.6]])
    B = numpy.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
    P = [0.2, 0.3, 0.5]
    one = dispersed_hmm()
    one.fit(A, B, P)
    print('\n\n前向后向　　:  ', one.forward_back(
        [0, 1, 0, 0, 1, 0, 1, 1], 4 - 1, 3 - 1))
    print('预测序列(从　0 数起)　　:  ', one.decode([0, 1, 0, 1], algorithm='viterbi'))

    print('\n\n经过验证, 以上结果与　hmmlearn 包所得结果一致')
