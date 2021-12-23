# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from scipy import special as sci
from sklearn.model_selection import train_test_split
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

 
 
#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    n, m = x_train.shape
    w = np.mat(np.eye(n)) #identity matrix 
    iden = np.mat(np.eye(m))

    test_datum = test_datum.reshape(test_datum.shape[0], 1)
    y_train = y_train.reshape(y_train.shape[0], 1)

    tau_denom = -2*tau*tau
    denom = np.exp(sci.logsumexp(l2(test_datum.T, x_train)/tau_denom)) #weighting over all examples

    for i in range(n):
      xi = x_train[i].reshape(1, m)
      w[i, i] = np.exp(sci.logsumexp(l2(test_datum.T, xi)/tau_denom))/denom #numerator
     
    for i in range(m):
      iden[i, i] = lam

    A = np.matmul(np.matmul(x_train.T, w), x_train) + iden #X.T*A*X+lam*I
    c = np.matmul(np.matmul(x_train.T, w), y_train) #X.T*A*y    

    theta = np.linalg.solve(A, c)

    y_hat = np.dot(test_datum.T, theta)
    return y_hat[0,0] #b/c y_hat is a 1x1 matrix




def run_validation(x,y,taus,val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''
    validation_loss = []
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.3)
    n = x_val.shape[0]

    for tau in taus:
      loss = 0
      for i in range(n):
        y_hat = LRLS(x_val[i], x_val, y_val, tau)
        loss += (y_hat - y_train[i])**2
      loss = loss/n
      validation_loss.append(loss)
    return validation_loss


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200)
    test_losses = run_validation(x,y,taus,val_frac=0.3)
    plt.semilogx(taus, test_losses)
    plt.xlabel("Taus")
    plt.ylabel("Average loss")
    plt.title("Taus vs Average Loss")
    plt.show()

