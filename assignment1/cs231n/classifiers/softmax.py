# coding=UTF-8
import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:SGD
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

    # 生成W一样shape的全0矩阵
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # 得分
    scores = np.dot(X, W)
    # 转换成指数
    exp_scores = np.exp(scores)
    exp_scores_sum = np.sum(exp_scores, axis=1)
    # 标签值
    exp_scores_y = exp_scores[range(scores.shape[0]), y]

    # 求总和
    exp_sum = np.sum(exp_scores)
    # 归一化
    normalize_scores = exp_scores_y / exp_scores_sum
    Li = -np.log(normalize_scores)

    # 对Li 求平均
    Li_mean = np.mean(Li)
    loss = Li_mean + 0.5 * reg * np.sum(W * W)

    #
    # scores = np.dot(X, W)
    # e_scores = np.exp(scores)
    # e_scores_sum = np.sum(e_scores, axis=1)
    # # e_scores[range(500),y]
    # # e_scores 对应的标签
    #
    # e_scores_yi = e_scores[range(scores.shape[0]), y]
    # L_i = - np.log(e_scores_yi / e_scores_sum)
    #
    # L = np.mean(L_i)
    # loss = L + 0.5 * reg * np.sum(W * W)
    #

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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################


    # 先求W*X
    # f_yi是xi对应最高分的分类的score值，
    W = W.T
    X = X.T

    # get dimensions
    (C, D) = W.shape
    N = X.shape[1]


    scores = np.dot(W, X)
    # 数据的稳定性：numeric instability
    scores -= np.max(scores)



    # numeric instability:数值不稳定
    # vectorized: Li = -f_yi+log(sum(exp(f_j))) 不求exp再求log，而是做一个化简

    # 构造对应的y矩阵，C行，每一列对应一个输入xi,对应的label为1，其他为0
    y_matrix = np.zeros(shape=(C, N))
    y_matrix[y, range(N)] = 1

    #loss的第一项 −f_yi
    loss_1 = -scores[y, range(N)]

    #loss的第二项 log(sum(exp(f_j)))
    loss_2 = np.log(np.sum(np.exp(scores),axis=0))

    loss= np.mean(loss_1 + loss_2)
    loss += 0.5 * reg * np.sum(W*W)

    # print loss


    # 下面开始求梯度
    # 对W求导，结果与W同样的维度 D*C,也就是X.T * (X * W) X.T是DxN, X*W是NxC 结果就是DxC,这里已经对X和W做了转置
    # dW = X * (WX).T
    sum_exp_scores = np.sum(np.exp(scores), axis=0)
    # 加上1e-8防止sum_exp_scores的某元素为0的情况
    # dl/dw = exp_fj/sum(exp_fk) * x
    sum_exp_scores = 1/(sum_exp_scores+1e-8)
    dW = np.multiply(np.exp(scores) , sum_exp_scores)

    # dW.shape (10L, 3073L)
    # X.shape (3073L, 500L)
    # y_matrix.shape (10L, 500L)
    dW = np.dot(dW, X.T)
    # 针对dL/dW_yi有
    # y_matrix.shape:(10L, 500L)
    dW -= np.dot(y_matrix, X.T)

    dW = dW/float(N)
    dW += reg*W
    dW = dW.T







    # y_mat = np.zeros(shape=(C, N))
    # y_mat[y, range(N)] = 1
    #
    # # matrix of all zeros except for a single wx + log C value in each column that corresponds to the
    # # quantity we need to subtract from each row of scores
    # correct_wx = np.multiply(y_mat, scores)
    #
    # # create a single row of the correct wx_y + log C values for each data point
    # sums_wy = np.sum(correct_wx, axis=0)  # sum over each column
    #
    # exp_scores = np.exp(scores)
    # sums_exp = np.sum(exp_scores, axis=0)  # sum over each column
    #
    # result = np.log(sums_exp)
    #
    # result -= sums_wy
    #
    # loss = np.sum(result)
    #
    # # Right now the loss is a sum over all training examples, but we want it
    # # to be an average instead so we divide by num_train.
    # loss /= float(N)
    #
    #
    # # Add regularization to the loss.
    # loss += 0.5 * reg * np.sum(W * W)
    #
    # sum_exp_scores = np.sum(exp_scores, axis=0)  # sum over columns
    # sum_exp_scores = 1.0 / (sum_exp_scores + 1e-8)
    #
    # dW = exp_scores * sum_exp_scores
    #
    #
    # dW = np.dot(dW, X.T)
    # dW -= np.dot(y_mat, X.T)
    # dW /= float(N)
    # # Add regularization to the gradient
    # dW += reg * W
    # dW = dW.T
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    return loss, dW
