# the structure of neural network: 
#    input layer with 2 inputs
#    1 hidden layer with 2 units, tanh()
#    output layer with 1 unit, sigmoid()

import numpy as np
import math
import scipy
from scipy.special import expit

def loadData(filename):
  """
  Brief : 
    Opens file "filename" and load dataset from file and output as an array
  
  Parameters :
    filename : Name of the file.
  
  Returns : 
    An array containing dataset in the file.
  """
  # Store dataset
  X = []

  # Open filename.txt
  file = open(filename)

  # Read all lines
  lines = file.readlines()

  # for each line in file, append each data into X
  i = 0
  for line in lines:
    X.append([])
    words = line.split(" ")
    X[i].append(1)
    for word in words:
      X[i].append(int(word))
    i += 1
  
  return np.asarray(X)

def paraIni():
  """
  Brief : 
    Initialize the parameters of the artificial neural network.

  Returns : 
    A list of hidden layer and output layer weights for all nodes.
  """
  wh = 2 * np.random.random_sample((2,3)) - 1
  wo = 2 * np.random.random_sample((1,3)) - 1

  return [wh,wo]
  
def feedforward(X, W):
  """
  Brief : 
    Perform feed forward algorithm of ANN.

  Parameters : 
    X : Dataset.

    W : Weights.
  
  Returns : 
    A list containing [oh, ino, oo].
    oh - output result of hidden layer nodes.
    ino - input of the output layer nodes
    oo - output result of the output layer nodes.
  """
  # Use the 3 feature to form a X matrix
  X2 = np.asarray([x[:-1] for x in X])

  # oh = tanh(Wh * X^T)
  oh = np.tanh(np.dot(W[0], np.transpose(X2)))
  
  # ino = 1st row 1, ..., n. The rest is oh
  ino = np.vstack((np.full((1, oh.shape[1]), 1.0), oh))

  # oo = sigmoid(Wo * Ino)
  oo = expit(np.dot(W[1], ino))

  return [oh,ino,oo]
  
def errCompute(Y,Yhat):
  """
  Brief : 
    Evaluates the error of the model.

  Parameters : 
    Y : Actual result.

    YHat : Model predicted result.
  
  Returns : 
    The error of the model.
  """
  # E = (1/2m) * summation((y-yhat)^2)
  J = np.sum([(a - b)**2 for a,b in zip(Y,np.transpose(Yhat))]) / (2.0 * Y.shape[0])
  return J

def backpropagate(X, W, intermRslt, alpha):
  """
  Brief : 
    Perform back propagation algorithm with gradient descent method.

  Parameters : 
    X : Dataset.

    W : Weights.

    intermRslt : Feed forward output.

    alpha : Learning rate.
  
  Returns : 
    The list containing the learned parameters.
  """
  wh = np.copy(W[0])
  wo = np.copy(W[1])

  # backpropation
  # for each tuple in dataset
  oo = np.copy(intermRslt[-1])
  yt = np.transpose(X[:,-1:])
  delta_o = (yt - oo) * oo * (1.0 - oo)                     # find delta of all output layer nodes
  oh = np.copy(intermRslt[0])
  wo_primet = np.transpose(np.copy(wo[:,1:]))
  delta_h = np.matmul(wo_primet, delta_o) * (1.0 - oh * oh) # Find delta of hidden layer node

  # Update the weight values
  wo += alpha * np.matmul(delta_o, np.transpose(intermRslt[1])) / X.shape[0]
  wh += alpha * np.matmul(delta_h, X[:,:-1]) / X.shape[0]

  return [wh,wo]
