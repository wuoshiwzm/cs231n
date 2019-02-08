# coding=UTF-8
import numpy as np

"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    next_h = np.tanh(np.dot(prev_h,Wh)+np.dot(x,Wx)+b)  #(1,H) (N,H)
    cache = {
        'prev_h': prev_h,
        'x': x,
        'Wx': Wx,
        'Wh': Wh,
        'b': b,
        'next_h': next_h,
    }




    # next_h = np.tanh(np.dot(prev_h, Wh) + np.dot(x, Wx) + b)
    # cache = {
    #     'prev_h': prev_h,
    #     'x': x,
    #     'Wx': Wx,
    #     'Wh': Wh,
    #     'b': b,
    #     'next_h': next_h,
    # }

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (N, H) 这里应该是(D,H)?
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################

    # 'prev_h': prev_h,
    # 'x': x,
    # 'Wx': Wx,
    # 'Wh': Wh,
    # 'b': b,
    # 'next_h': next_h,

    x= cache['x']
    Wx = cache['Wx']
    Wh = cache['Wh']
    b= cache['b']
    prev_h = cache['prev_h']
    next_h = cache['next_h']


    dh_raw = dnext_h*(1-next_h*next_h)    #(N,H)  h_raw  指的是 np.dot(prev_h, Wh) + np.dot(x, Wx) + b
    dx = np.dot(dh_raw, Wx.T)   #
    dprev_h = np.dot(dh_raw, Wh.T)
    dWx = np.dot(x.T, dh_raw)  #(D,H)
    dWh = np.dot(prev_h.T, dh_raw)   #(H, H)
    db = np.sum(dh_raw,axis=0)



    #
    # x = cache['x']
    # prev_h = cache['prev_h']
    # Wx = cache['Wx']
    # Wh = cache['Wh']
    # b = cache['b']
    # next_h = cache['next_h']
    #
    # daffine_output = dnext_h * (1 - next_h * next_h)
    # dx = daffine_output.dot(Wx.T)
    # dprev_h = daffine_output.dot(Wh.T)
    # dWx = x.T.dot(daffine_output)
    # dWh = prev_h.T.dot(daffine_output)
    # db = np.sum(dnext_h * (1 - next_h * next_h), axis=0)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above.                                                                     #
    ##############################################################################
    N,T,D = x.shape
    _,H = h0.shape
    h = np.zeros((N,T,H))
    cache = {}
    # 一个t对应一个词
    for t in range(T):
        if t == 0:
            h[:,t,:], cache[t] = rnn_step_forward(x[:,t,:], h0, Wx, Wh, b)
        else:
            h[:,t,:], cache[t] = rnn_step_forward(x[:,t,:], h[:,t-1,:], Wx, Wh, b)



    # code from internet
    # N, T, D = x.shape
    # _, H = h0.shape
    # h = np.zeros((N, T, H))
    # cache = {}
    # for t in range(T):
    #     if t == 0:
    #         h[:, t, :], cache[t] = rnn_step_forward(x[:, t, :], h0, Wx, Wh, b)
    #     else:
    #         h[:, t, :], cache[t] = rnn_step_forward(x[:, t, :], h[:, t - 1, :], Wx, Wh, b)





    # code from liuweijie
    # N, T, D = x.shape
    # H = b.shape[0]
    #
    # h = np.zeros((N, T+1, H), dtype=np.float32)
    # h[:, 0, :] = h0
    # cache = []
    # for i in range(N):
    #
    #   betch_cache = []
    #   for t in range(T):  # forward each time
    #     h[i, t+1, :], cache_temp = rnn_step_forward(x[i, t, :], h[i, t, :], Wx, Wh, b)
    #     betch_cache.append(cache_temp)
    #
    #   cache.append(betch_cache)
    #
    # h = h[:, 1:, :]
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above.                                                             #
    ##############################################################################
    N,T,H = dh.shape
    x = cache[T-1]['x']
    prev_h = cache[T-1]['prev_h']
    Wx = cache[T-1]['Wx']
    Wh = cache[T-1]['Wh']
    next_h = cache[T-1]['next_h']
    _,D = x.shape
    dx = np.zeros((N,T,D))
    dWx = np.zeros(Wx.shape)  #(D,H)
    dWh = np.zeros(Wh.shape)  #(H,H)
    db = np.zeros((H))
    dprev_h = np.zeros(prev_h.shape) #(N,H)

    for t in range(T-1,-1,-1):
        dx[:,t,:], dprev_h, dWx_add, dWh_add, db_add =rnn_step_backward(dh[:,t,:]+dprev_h, cache[t])  # 注意这里的dprev_h要加上dh，也就是他的后一个h
        dWx +=  dWx_add
        dWh +=  dWh_add
        db += db_add
    dh0 = dprev_h

    # (N, T, H) = dh.shape
    # x = cache[T - 1]['x']
    # prev_h = cache[T - 1]['prev_h']
    # Wx = cache[T - 1]['Wx']
    # Wh = cache[T - 1]['Wh']
    # next_h = cache[T - 1]['next_h']
    # N, D = x.shape
    # dx = np.zeros((N, T, D))
    # dWx = np.zeros(Wx.shape)
    # dWh = np.zeros(Wh.shape)
    # db = np.zeros((H))
    # dprev = np.zeros(prev_h.shape)
    #
    # for t in range(T - 1, -1, -1):
    #     dx[:, t, :], dprev, dWx_local, dWh_local, db_local = rnn_step_backward(dh[:, t, :] + dprev, cache[t])
    #     dWx += dWx_local
    #     dWh += dWh_local
    #     db += db_local
    #
    # dh0 = dprev
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
	把词转化为向量
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.
    每个词对应D维向量
    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x must be in the range 0 <= idx < V.            
    - W: Weight matrix of shape (V, D) giving word vectors for all words.所以的词向量

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This should be very simple.                                          #
    ##############################################################################
    V,D = W.shape
    N,T = x.shape
    out = np.zeros((N,T,D))

    for t in range(T):
        for n in range(N):
            out[n,t] = W[x[n,t]]   #每一个out[n,t] 对应一个向量W[x[n,t]][:] , 即对应W中的一行


    cache = {
        'x':x,
        'W':W,
        'V':V,
        'D':D
    }

    # V, D = W.shape
    # N, T = x.shape
    # out = np.zeros((N, T, D))
    # for n in range(N):
    #     for t in range(T):
    #         out[n, t] = W[x[n, t]]
    #
    # cache = {
    #     'x': x,
    #     'W': W,
    #     'V': V,
    #     'D': D,
    # }
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at 指定index处进行add操作

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    x = cache['x']
    W = cache['W']
    V = cache['V']
    D = cache['D']
    dW = np.zeros((V,D))
    np.add.at(dW,x,dout)  #在x对应的位置上赋值dout, 其他为0，表示无梯度

    # x = cache['x']
    # W = cache['W']
    # V = cache['V']
    # D = cache['D']
    # dW = np.zeros((V, D))
    # np.add.at(dW, x, dout)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    N,D = x.shape
    N,H = prev_h.shape
    a = np.dot(x, Wx) + np.dot(prev_h,Wh) +b
    i = sigmoid(a[:, 0:H])
    f = sigmoid(a[:, H:2*H])
    o = sigmoid(a[:, 2*H:3*H])
    g = np.tanh(a[:, 3 * H:4 * H])

    next_c = f * prev_c + i * g
    next_h = o * np.tanh(next_c)

    cache = (x, prev_h, prev_c, Wx, Wh, i, f, o, g, next_h, next_c)


    # N, D = x.shape
    # _, H = prev_h.shape
    #
    # a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
    # i = sigmoid(a[:, 0: H])
    # f = sigmoid(a[:, H: 2 * H])
    # o = sigmoid(a[:, 2 * H: 3 * H])
    # g = np.tanh(a[:, 3 * H: 4 * H])
    #
    # next_c = f * prev_c + i * g
    # next_h = o * np.tanh(next_c)
    #
    # cache = (x, prev_h, prev_c, Wx, Wh, i, f, o, g, next_h, next_c)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################

    (x, prev_h, prev_c, Wx, Wh, i, f, o, g, next_h, next_c) = cache
    dnext_c += o * (1-np.tanh(next_c)**2) * dnext_h    #dh * dc
    di = g* dnext_c
    df = prev_c * dnext_c
    do = np.tanh(next_c) * dnext_h
    dg = i * dnext_c
    dprev_c = f * dnext_c

    da = np.hstack((i*(1-i)*di,f*(1-f)*df,o*(1-o)*do,(1-g**2)*dg)) #da就是对i,f,o,g求导

    dx = da.dot(Wx.T)
    # a = np.dot(x, Wx) + np.dot(prev_h,Wh) +b
    dprev_h = da.dot(Wh.T)
    dWx = x.T.dot(da)
    dWh = prev_h.T.dot(da)
    db = np.sum(da,axis=0)





    # (x, prev_h, prev_c, Wx, Wh, i, f, o, g, next_h, next_c) = cache
    #
    # dnext_c = dnext_c + o * (1 - np.tanh(next_c) ** 2) * dnext_h
    # di = dnext_c * g
    # df = dnext_c * prev_c
    # do = dnext_h * np.tanh(next_c)
    # dg = dnext_c * i
    # dprev_c = f * dnext_c
    # da = np.hstack((i * (1 - i) * di, f * (1 - f) * df, o * (1 - o) * do, (1 - g ** 2) * dg))
    #
    # dx = da.dot(Wx.T)
    # dprev_h = da.dot(Wh.T)
    # dWx = x.T.dot(da)
    # dWh = prev_h.T.dot(da)
    # db = np.sum(da, axis=0)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################

    # lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)  return next_h, next_c, cache
    N,T,D = x.shape
    N,H = h0.shape
    h = np.zeros((N,T,H))
    c = np.zeros((N,T,H))
    c0 = np.zeros(())
    cache = {}

    for t in range(T):
        if t == 0:
            h[:,t,:] , c[:,t,:],cache[t] =lstm_step_forward(x[:,t,:], h0, c0, Wx, Wh, b)
        else :
            h[:, t, :], c[:, t, :], cache[t] = lstm_step_forward(x[:,t,:], h[:,t-1,:], c[:,t-1,:], Wx, Wh, b)

    return h, cache


    #
    # N, T, D = x.shape
    # _, H = h0.shape
    # h = np.zeros((N, T, H))
    # c = np.zeros((N, T, H))_
    # c0 = np.zeros((N, H))
    # cache = {}
    # for t in range(T):
    #     if t == 0:
    #         h[:, t, :], c[:, t, :], cache[t] = lstm_step_forward(x[:, t, :], h0, c0, Wx, Wh, b)
    #     else:
    #         h[:, t, :], c[:, t, :], cache[t] = lstm_step_forward(x[:, t, :], h[:, t - 1, :], c[:, t - 1, :], Wx, Wh, b)
    #
    # ##############################################################################
    # #                               END OF YOUR CODE                             #
    # ##############################################################################
    #
    # return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    # lstm_step_backward(dnext_h, dnext_c, cache) return dx, dprev_h, dprev_c, dWx, dWh, db
    (N,T,H) = dh.shape
    # start from the very last LSTM layer
    x, prev_h, prev_c, Wx, Wh, i, f, o, g, next_h, next_c = cache[T-1]

    N,D = x.shape
    dx = np.zeros((N,T,D))
    dWx = np.zeros((Wx.shape))
    dWh = np.zeros((Wh.shape))
    db = np.zeros((4*H))
    dprev_h = np.zeros(prev_h.shape)
    dprev_c = np.zeros(prev_c.shape)

    for t in range(T-1,-1,-1):
        # 这里的dprev_h实际上变成了下一轮的dnext_h
        dx[:,t,:], dprev_h, dprev_c, dWx_calc, dWh_calc, db_calc=lstm_step_backward(dh[:,t,:]+dprev_h, dprev_c, cache[t])
        dWh += dWh_calc
        dWx += dWx_calc
        db += db_calc

    dh0  = dprev_h
    #
    #
    # (N, T, H) = dh.shape
    # x, prev_h, prev_c, Wx, Wh, i, f, o, g, next_h, next_c = cache[T - 1]
    #
    # N, D = x.shape
    # dx = np.zeros((N, T, D))
    # dWx = np.zeros(Wx.shape)
    # dWh = np.zeros(Wh.shape)
    # db = np.zeros((4 * H))
    # dprev = np.zeros(prev_h.shape)
    # dprev_c = np.zeros(prev_c.shape)
    #
    # for t in range(T - 1, -1, -1):
    #     dx[:, t, :], dprev, dprev_c, dWx_local, dWh_local, db_local = lstm_step_backward(dh[:, t, :] + dprev, dprev_c,
    #                                                                                      cache[t])
    #     dWx += dWx_local
    #     dWh += dWh_local
    #     db += db_local
    #
    # dh0 = dprev
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.
    转换成word_embedding的格式
    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]  # 这里就是H
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b  #(N,T,M)  相当于 (N,T,H) ,Whh的shape
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    vocabulary:V
    minibatch:N
    x: score of V
    y:
    a cross-entropy loss: sum loss over all timesteps

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V 每个y对应V中的一行的index，也就是这行对应的词
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss. 是否计算loss值

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)  # x变换为V列的
    y_flat = y.reshape(N * T) #y变换为N行 1列的值（列值对应词库的index）(对应一个t)
    mask_flat = mask.reshape(N * T)

    # softmax 的 loss 计算
    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N

    # softmax 的求导。。。（前面已经计算过）
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print 'dx_flat: ', dx_flat.shape

    # 把求导值换成x的shape
    dx = dx_flat.reshape(N, T, V)

    return loss, dx
