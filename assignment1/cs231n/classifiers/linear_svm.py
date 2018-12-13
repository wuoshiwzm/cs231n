import numpy as np
from random import shuffle


def svm_loss_naive(W, X, y, reg):
    """
    SVM的损失函数计算
    Li = sum(i!=j)(max(0,sj-si+1))
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    D是每个图片的维度
    N是输入图片的个数
    C这里是10，表示每个类对应的得分
    注意这里是  X.dot(W)， 不是W.dot(X)
    reg 正则化
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data. 随机梯度下降
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    返回元组
    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # 初始化dW为0

    # compute the loss and the gradient
    # 计算损失与梯度
    num_classes = W.shape[1]  # 图片总的分类数
    num_train = X.shape[0]  # 总的图片数
    loss = 0.0
    # 遍历每一个训练图片
    for i in xrange(num_train):
        # X[i]表示第i个图片
        scores = X[i].dot(W)  # 维度为 N*C
        for j in xrange(num_classes):

            # SVM的损失函数计算
            # Li = sum(y(i)!=j)(max(0,s(j)-s(yi)+1))
            if j != y[i]:
                margin = scores[j] - scores[y[i]] + 1  # note delta = 1
                if margin > 0:
                    loss += margin

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    # 再求loss的平均值
    loss /= num_train

    # Add regularization to the loss.
    # y = wx +b
    # b = 1/2(reg)(W^2)
    # 正则化项的表达为 lambda * R(W)
    # 这里用L2正则化项  np.sum(W * W)
    loss += 0.5 * reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    return loss, dW


# 损失函数的向量化
def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################

    num_labels = W.shape[1]  # 图片总的分类数
    num_pics = X.shape[0]  # 总的图片数

    # w*x
    X_times_W = np.dot(X, W)

    # 构造一个对角线是1,其他全是1的矩阵 C*C的矩阵
    cal = np.eye(num_labels,dtype=int)
    cal = cal-1
    cal = cal + np.eye(num_labels,dtype=int)

    # svm的Li损失函数 Li = (1/num_pics)sum(y(i)!=j)(max(0,s(j)-s(yi)+1))
    scores = X_times_W #N*C维  scores[0] ~ scores[N]

    row_max =scores.dot(cal)
    Li = np.sum(np.max(scores,0),axis=0)
    print scores.shape






    # b = lambda * R(W)
    # 求R(W)
    b = 0.5 * reg * np.sum(W * W)














    num_features, num_classes = W.shape
    num_train = X.shape[0]

    XW = np.dot(X, W)

    Y_true = XW[np.arange(num_train), y]

    margin = XW.T - Y_true + 1
    margin = margin.T
    margin[np.arange(num_train), y] = 0

    margin = np.maximum(margin, np.zeros((num_train, num_classes)))
    loss = np.sum(margin)

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################

    # Binarize into integers
    binary = margin
    binary[margin > 0] = 1

    # Perform the two operations simultaneously
    # (1) for all j: dW[j,:] = sum_{i, j produces positive margin with i} X[:,i].T
    # (2) for all i: dW[y[i],:] = sum_{j != y_i, j produces positive margin with i} -X[:,i].T
    col_sum = np.sum(binary, axis=1)
    binary[np.arange(num_train), y] = -col_sum[range(num_train)]
    dW = np.dot(X.T, binary)

    # Divide
    dW /= num_train

    # Regularize
    dW += reg * W

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
