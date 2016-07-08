import random
import numpy as np
from utils import *
import sys

def loadDataset(filename):
    fp = open(filename)
    numFeats = len(fp.readline().split())-1
    data_x = []
    data_y = []
    for l in fp:
        row = []
        for unit in l.split():
            row.append(float(unit))
        data_x.append(row[0:numFeats])
        data_y.append([row[-1]])
    return data_x,data_y

def getMinMax(mat):
    n = len(mat)
    m = 0
    for k in mat[0]:
        m=m+1
    MinNum = [999999999]*m
    MaxNum = [0]*m
    for i in mat:
        for j in range(0,m):
            if i[j] > MaxNum[j]:
                MaxNum[j] = i[j]
    for p in mat:
        for q in range(0,m):
            if p[q] <= MinNum[q]:
                MinNum[q] = p[q]
    return MinNum,MaxNum

def autoNorm(mat,MinNum,MaxNum):
    #MinNum, MaxNum = getMinMax(mat)
    section = list(map(lambda x: x[0]-x[1],zip(MaxNum,MinNum)))
    NormMat=[]

    for kk in mat:
        distance=list(map(lambda x: x[0]-x[1],zip(kk,MinNum)))
        value=list(map(lambda x: x[0]/x[1],zip(distance,section)))
        NormMat.append(value)
    return NormMat

class Network(object):
    def __init__(self,sizes,activation):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]#randn return a sample from standard normal distribution
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
        print self.weights
        self.activation = activation
        if activation == tanh:
            self.dactivation = dtanh
        if activation == sigmoid:
            self.dactivation = dsigmoid

    def feedforward(self,a):#multiple layers to predict regression problem, last layer none activation
        for b,w in zip(self.biases[:-1],self.weights):
            a = activation(np.dot(w,a) + b)

        a = np.dot(self.weights[-1],a) + self.biases[-1]
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta):
        #mini_batch_size the min number of training batch
        
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)#shuffle the data
            
            mini_batchs = [training_data[k:k+mini_batch_size] for k in xrange(0,n,mini_batch_size)]
            for mini_batch in mini_batchs:
                self.update_mini_batch(mini_batch, eta)
            
            print 'Epoch {0} complete'.format(j)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

            self.weights = [w - (eta/len(mini_batch)) * nw for w,nw in zip(self.weights, nabla_w)]
            self.biases = [b - (eta/len(mini_batch)) * nb for b, nb in zip(self.biases,nabla_b)]
            print "weights"+str(self.weights)
            print "biases"+str(self.biases)

    def backprop(self,x,y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = numpy.array([x]).T
        activations = []
        activations.append(activation)
        zs = []

        #hidden layer
        for b,w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activation(z)
            activations.append(activation)
        
        output_r = np.dot(self.weights[-1],activations[-1]) + self.biases[-1]

        print "y"+str(y)
        print "output_r"+str(output_r)
        zs.append(output_r)
        activations.append(output_r)

        delta = output_r - y
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = self.dactivation(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        
        return (nabla_b,nabla_w)

if __name__ == "__main__":
    data_x,data_y = loadDataset(sys.argv[1])
    minNum_x,maxNum_x = getMinMax(data_x)
    data_x = autoNorm(data_x,minNum_x,maxNum_x)
    minNum_y,maxNum_y = getMinMax(data_y)
    data_y = autoNorm(data_y,minNum_y,maxNum_y)

    data_xx = np.array(data_x)
    data_yy = np.array(data_y*100)
    training_data = zip(data_xx, data_yy)
    sizes=[2,3,1]
    #net = Network(sizes,tanh)
    net = Network(sizes,sigmoid)
    net.SGD(training_data,20,10,0.0001)
























        



















