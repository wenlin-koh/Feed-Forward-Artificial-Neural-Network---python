import matplotlib.pyplot as plt
from annxor import *

def FFMain(filename,numIteration, alpha, W):
  # Data load
  X = loadData(filename)

  # Weight Initialization
  # W = paraIni()
  
  # Number of features
  n = X.shape[1]
  
  # Error
  errHistory = np.zeros((numIteration,1))

  for i in range(numIteration):
    #feedforward
    intermRslt=feedforward(X,W)
    #Cost function
    errHistory[i,0]=errCompute(X[:,n-1:n],intermRslt[2])
    #backpropagate
    W=backpropagate(X,W,intermRslt,alpha)

  Yhat=np.around(intermRslt[2]) 
  return [errHistory,intermRslt[2],W]

def TestBaseMain(filename, numIteration, alpha):
  # Test base load
  X = loadData(filename)
  print(X.shape)
  
  # Test feed forward
  wh = np.array([[0.1859,-0.7706,0.6257],[-0.7984,0.5607,0.2109]])
  wo = np.array([[0.1328,0.5951,0.3433]])
  W = [wh, wo]
  intermRslt = feedforward(X, W)
  print(intermRslt[-1])

  err = errCompute(X[:, -1:], intermRslt[2])
  print(err)

  W = backpropagate(X, W, intermRslt, alpha)
  print(W[0])
  print(W[1])
  
  errHistory = np.zeros((numIteration,1))
  # Number of features
  n = X.shape[1]

  for i in range(numIteration - 1):
    #feedforward
    intermRslt=feedforward(X,W)
    #Cost function
    errHistory[i,0]=errCompute(X[:,n-1:n],intermRslt[2])
    #backpropagate
    W=backpropagate(X,W,intermRslt,alpha)

  Yhat = np.around(intermRslt[2])
  print(intermRslt[2])

# TestBaseMain("XOR.txt", 10000, 0.5)

numItr = [100, 1000, 5000, 10000]
alphas = [0.01, 0.5]
W = paraIni()

for a in alphas:
  for itr in numItr:
    R = FFMain("XOR.txt", itr, a, W)
    x = [i for i in range(itr)]
    y = [float(a) for a in R[0]]
    plt.figure()
    plt.title('Curve of Error Function (alpha = {}, itr = {})'.format(a, itr))
    plt.plot(x, y)
    plt.xlabel("Iteration #")
    plt.ylabel("Error")
    # plt.savefig()
    plt.show()
    print(R[0][0])

