import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  N, D = X.shape
  _, C = W.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  for i in range(N):
    y_hat = np.dot(X[i,:], W)
    y_hat -= np.max(y_hat)
    log_probs = np.exp(y_hat)/np.sum(np.exp(y_hat))
    loss -= np.log(log_probs[y[i]])
    dscores = log_probs

    for j in range(C):

      if j == y[i]:
        dW[:, j] += -X[i,:].T + X[i,:].T*dscores[j]
      else:
        dW[:, j] += X[i,:].T*dscores[j]

  dW = dW/N + reg * W
  loss /= N
  loss += reg*np.sum(W*W)

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  N, D = X.shape
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  y_hat = np.dot(X, W)
  y_hat -= np.max(y_hat, axis = 1).reshape(-1,1)
  log_probs = np.exp(y_hat)/np.sum(np.exp(y_hat), axis = 1).reshape(-1,1)

  loss = np.mean(-np.log(log_probs[range(N), list(y)])) + reg*np.sum(W*W)
  grad = log_probs
  grad[range(N), list(y)] -= 1
  dW = np.dot(X.T, grad)/N + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

