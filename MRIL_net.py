import numpy as np
from mnist.loader import MNIST

def loadImage(): #E
    #read mnist image file and save them to 28*28 matrix, shape = (j, i)
    mndata = MNIST('samples')
    images, labels = mndata.load_training()
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

sigmoidChoice = 1
exChoice = 3
inputMatrices, labels = loadImage()
inputX, inputY = inputMatrices.shape #j, i
dendriteX = inputX  #j
dendriteY = 10  #k
somaX = 10  #m
somaY = dendriteY  #k
W = np.random.randn(inputY, dendriteY) #weight: shape = (i, k)
G = np.random.randn(somaY, somaY) #lateral inhibition weight: shape = (k, k)
G = np.triu(G, k=0)
G += G.T - np.diag(G.diagonal())
np.fill_diagonal(G, 0) #set diagonal to 0

def sigmoid(x): #sigmoid function
    if sigmoidChoice == 1:
        return 1/(1+np.exp(-x))   # formula 11
    elif sigmoidChoice == 2:
        return np.tanh(x) # formula 12
    else:
        assert False, "Sigmoid function error"
        
def sigmoidDerivative(x): #sigmoid function derivative
    if sigmoidChoice == 1:
        return np.exp(-x)/(1+np.exp(-x))**2 # formula 13
    elif sigmoidChoice == 2:
        return 1 - np.tanh(x)**2 # formula 14
    else:
        assert False, "Sigmoid function derivative error"

def J(x, y): #exponential decay function
    sigma = 1
    if exChoice == 1:
        return np.exp((-(x-y)**2)/(2*sigma**2)) #formula 7
    elif exChoice == 2:
        return np.exp((-np.abs(x-y))/(2*sigma)) #formula 8
    elif exChoice == 3:
        return 1/(1+np.abs(x-y)) #formula 9
    elif exChoice == 4:
        return 1/(1+(x-y)**2) #formula 10
    else:
        assert False, "Exponential decay function error"

def DendriteProcess(input): #V
    # build a dendritic layer: shape = (j, k) 
    # transfer E to V
    dendriteMat = np.dot(input, W) #formula 1
    return dendriteMat
  
def SomaProcess(input): #U
    # build a soma layer: shape = (m, k)
    # transfer V to U
    # build a matrix for J(x,y)
    J_jm = np.zeros((dendriteX, somaX))
    for i in range(dendriteX):
        for j in range(somaX):
            J_jm[i][j] = J(i, j)
    somaMat = np.dot(J_jm.T, input)
    J_mm = np.zeros((dendriteX, somaX))
    for i in range(somaX):
        for j in range(somaX):
            J_mm[i][j] = J(i, j)
    somaMat = somaMat - np.dot(J_mm.T, np.dot(sigmoid(somaMat), G)) #formula 2
    return somaMat
           
class Neuron():
    def __init__(self):
        self.lr_w = 0.01
        self.lr_g = 0.01

    def update(self, dW, dG):
        # update W
        alpha = 0.01
        W = W + self.lr_w * alpha * self.dW #formula 5
        # update G
        beta = 0.01
        G = G + self.lr_g * beta * self.dG #formula 6
        
    def train(self):
        for p in range(inputMatrices.size()):
            inputMat = inputMatrices[p]
            dend = DendriteProcess(inputMat)
            soma = SomaProcess(dend)
            dW = np.zeros((inputY, dendriteY))
            for i in range(inputY): #i
                for k in range(dendriteY): #k
                    for m in range(somaX): #m
                        dW[i][k] += sigmoidDerivative(-dend[m][k]) * (sigmoid(soma[m][k]-dend[m][k])) * inputMat[m][i] #formula 3
            dW /= somaX
            dG = np.zeros((dendriteY, dendriteY))
            for i in range(dendriteY): #k'
                for j in range(dendriteY): #k
                    if i == j:
                        dG[i][j] = 0
                    else:
                        c = 0
                        for m in range(somaX): #m
                            a = 0
                            b = 0
                            for n in range(somaX): #m'
                                if m != n:
                                    a += sigmoid(soma[n][j])
                                    b += sigmoid(soma[n][i])
                            a/=(somaX-1)
                            b/=(somaX-1)
                            c += (sigmoid(soma[m][j]) - a) * (sigmoid(soma[m][i]) - b)
                        dG[i][j] = c/somaX #formula 4
            self.update(dW, dG)