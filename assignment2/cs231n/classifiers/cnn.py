# coding=UTF-8
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               num_filters_2=32,filter_size_2=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.

    conv-> affine_hidden -> out
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    conv_param_2 = {'stride': 1, 'pad': (filter_size - 1) / 2}
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    dropout_param = {'p':0.5,'mode':'train'}
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    # 初始化权重W1,W2,W3, 和bias : b1,b2,b3
    C,H,W = input_dim
    # W:F,C,H_filter,W_filter
    # b:F,
    self.params['W1'] = weight_scale * np.random.randn(num_filters,C,filter_size,filter_size)
    self.params['b1'] = np.zeros((num_filters,))
    self.params['conv_param'] = conv_param
    self.params['conv_param_2'] = conv_param_2
    self.params['pool_param'] = pool_param
    self.params['dropout_param'] = dropout_param

    # gamma beta initial
    self.params['gamma'], self.params['beta'] = np.ones(num_filters), np.zeros(num_filters)

    self.params['bn_param1'] = {
      'mode': 'train'
    }
    self.params['bn_param2'] = {
      'mode': 'train'
    }

    # 每个卷积层的输出shape
    pad = conv_param['pad']
    conv_stride = conv_param['stride']
    conv_H = int(1 + (H+2*pad - filter_size)/conv_stride)
    conv_W = int(1+(W+2*pad-filter_size)/conv_stride)

    # 池化层的输出shape(此处用max_pool)
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    pool_stride = pool_param['stride']
    pool_H = int(1 + (conv_H - pool_height) / pool_stride)
    pool_W = int(1 + (conv_W - pool_width) / pool_stride)


    # 这里再加一个conv层
    # 上一层到这里的输出为 conv_out: (batch_size,num_filters,pool_H,pool_W)
    # conv_forward_strides(x, w, b, conv_param)

    # self.params['W1'] = weight_scale * np.random.randn(num_filters,C,filter_size,filter_size)
    self.params['W_2'] = weight_scale * np.random.randn(num_filters_2, num_filters, filter_size_2, filter_size_2)
    self.params['b_2'] = np.zeros((num_filters_2,))
    pad_2 = conv_param_2['pad']
    conv_stride_2 = conv_param_2['stride']

    conv_H_2 = int(1 + (pool_H+2*pad_2 - filter_size_2)/conv_stride_2)
    conv_W_2 = int(1 + (pool_W+2*pad_2 - filter_size_2)/conv_stride_2)



    # 每一个卷积层经过池化输出 pool_H*pool_W，一共有num_filters个卷积核，
    # 所以最终输出给下层的数据维度为[num_filters,pool_H,pool_W]
    # 这里直接转化为1维,最终输出为hidden_dim:隐藏层中的神经元个数
    # num_input = num_filters * pool_H * pool_W
    num_input = num_filters_2 * conv_H_2 * conv_W_2 #这里是更新为新加卷积层的输出
    self.params['W2'] = weight_scale * np.random.randn(num_input,hidden_dim)
    self.params['b2'] = np.zeros((hidden_dim,))
    self.params['gamma2'], self.params['beta2'] = np.ones(hidden_dim), np.zeros(hidden_dim)

    # 第三层 最终把隐层中的数据输出为分类
    num_input = hidden_dim
    num_out = num_classes
    self.params['W3'] = weight_scale * np.random.randn(num_input,num_out)
    self.params['b3'] = np.zeros((num_out,))
    # self.params['gamma3'], self.params['beta3'] = np.ones(num_out), np.zeros(num_out)


    # C, H, W = input_dim
    #
    # # params for conv_pool layer
    # self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
    # self.params['b1'] = np.zeros((num_filters,))
    # self.params['conv_param'] = conv_param
    # self.params['pool_param'] = pool_param
    #
    # H_conv_o = int(1 + (H + 2 * conv_param['pad'] - filter_size) / conv_param['stride'])
    # W_conv_o = int(1 + (W + 2 * conv_param['pad'] - filter_size) / conv_param['stride'])
    # H_pool_o = int(1 + (H_conv_o - pool_param['pool_height']) / pool_param['stride'])
    # W_pool_o = int(1 + (W_conv_o - pool_param['pool_width']) / pool_param['stride'])
    #
    # # params for the second layer - affine
    # num_input = num_filters * H_pool_o * W_pool_o
    # self.params['W2'] = weight_scale * np.random.randn(num_input, hidden_dim)
    # self.params['b2'] = np.zeros((hidden_dim,))
    #
    # # params for the third layer - affine
    # num_input = hidden_dim
    # num_output = num_classes
    # self.params['W3'] = weight_scale * np.random.randn(num_input, num_output)
    # self.params['b3'] = np.zeros((num_output,))

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      if isinstance(v, np.ndarray):
        self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W_2, b_2 = self.params['W_2'], self.params['b_2']

    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    gamma, beta = self.params['gamma'], self.params['beta']
    gamma2, beta2 = self.params['gamma2'], self.params['beta2']
    bn_param1 = self.params['bn_param1']
    bn_param2 = self.params['bn_param2']
    dropout_param = self.params['dropout_param']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = self.params['conv_param']
    conv_param_2 = self.params['conv_param_2']

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = self.params['pool_param']

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    N,C,H,W = X.shape


    # conv_relu_pool_forward -> batchnorm
    conv_out,conv_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param, dropout_param)
    # print conv_out.shape
    conv_out,batch_norm_cache = spatial_batchnorm_forward(conv_out, gamma, beta, bn_param1)
    # conv_out: (batch_size,C,conv_H,conv_W)

    # conv->relu
    # 这里再加一层卷积层  conv_forward_strides(x, w, b, conv_param): return out, cache
    conv_out_2,conv_cache_2 = conv_forward_fast(conv_out, W_2, b_2, conv_param_2)

    # print conv_out.shape
    affine_relu_out,affine_cache = affine_relu_forward(conv_out_2,W2,b2)
    # print conv_out.shape,affine_out.shape  (N,hidden_dims)

    affine_out,bn2_cache = batchnorm_forward(affine_relu_out, gamma2, beta2, bn_param2)
     # print affine_out.shape

    out,cache = affine_forward(affine_out,W3,b3)

    scores = out


    # Y_1, cache_1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    # Y_2, cache_2 = affine_relu_forward(Y_1, W2, b2)
    # Y_3, cache_3 = affine_forward(Y_2, W3, b3)
    # scores = Y_3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################

    # def softmax_loss(x, y):  return loss, dx
    #
    loss,dscores = softmax_loss(scores,y)
    loss += 0.5 * self.reg *(np.sum(W1*W1) +np.sum(W2*W2)+np.sum(W3*W3))


    dx3,dW3,db3 = affine_backward(dscores,cache)
    dW3 += self.reg*W3

    dx2, dgamma2, dbeta2 = batchnorm_backward(dx3, bn2_cache)
    dgamma2 += self.reg * gamma2
    dbeta2 += self.reg * beta2
    dx2,dW2,db2 = affine_relu_backward(dx2,affine_cache)
    dW2 += self.reg * W2


    dx_2,dW_2,db_2 = conv_backward_strides(dx2, conv_cache_2)
    dW_2 += self.reg * W_2

    # spatial_batchnorm_backward(dout, cache) return dx, dgamma, dbeta
    dx2, dgamma, dbeta = spatial_batchnorm_backward(dx_2, batch_norm_cache)
    dgamma += self.reg * gamma
    dbeta += self.reg * beta

    dx1,dW1,db1 = conv_relu_pool_backward(dx2,conv_cache)
    dW1 += self.reg * W1




    grads['W1'] = dW1
    grads['W_2'] = dW_2
    grads['W2'] = dW2
    grads['W3'] = dW3
    grads['b1'] = db1
    grads['b_2'] = db_2
    grads['b2'] = db2
    grads['b3'] = db3
    grads['gamma'] = dgamma
    grads['beta'] = dbeta
    grads['gamma2'] = dgamma2
    grads['beta2'] = dbeta2





    # loss, dy = softmax_loss(scores, y)
    # loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))
    #
    # dx3, dW3, db3 = affine_backward(dy, cache_3)
    # dW3 += self.reg * W3
    # dx2, dW2, db2 = affine_relu_backward(dx3, cache_2)
    # dW2 += self.reg * W2
    # dx1, dW1, db1 = conv_relu_pool_backward(dx2, cache_1)
    # dW1 += self.reg * W1
    #
    # grads['W1'] = dW1
    # grads['W2'] = dW2
    # grads['W3'] = dW3
    # grads['b1'] = db1
    # grads['b2'] = db2
    # grads['b3'] = db3

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
  
  
pass
