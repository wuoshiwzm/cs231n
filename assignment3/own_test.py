# coding=UTF-8
import numpy as np

# data I/O
data = open('input.txt', 'r').read()  # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
# 生成字符:index   和  index:字符的映射
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
# char_to_ix = { ch:i for i,ch in enumerate(chars) }
# ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 100  # size of hidden layer of neurons
seq_length = 25  # number of steps to unroll the RNN for  每隔25次计算一轮反向传播
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # input to hidden
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size) * 0.01  # hidden to output
bh = np.zeros((hidden_size, 1))  # hidden bias
by = np.zeros((vocab_size, 1))  # output bias


# x:(v,1) hs:(h,1) ys:(v,1)
# Wxh:(h,v) Whh:(h,h) Why:(v,h)


def lossFun(inputs, targets, hprev):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """

    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)  # hidden state
    loss = 0

    # forward pass 前向
    for t in xrange(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))  # 输入为 one hot格式,x[t]长度对应所有字母长度
        xs[t][inputs[t]] = 1  # (v,1)

        # update hidden state
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t - 1]) + bh)  # (h,1)

        # update output
        ys[t] = np.dot(Why, hs[t]) + by  # (v,1)

        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # scalar

        # total loss value equals the sum of every single module loss
        # here we use softmax loss(cross-entry)
        loss += -np.log(ps[t][targets[t], 0])  # targets 是给定的下一个字符的位置，拿到这个位置对应的softmax损失值

    # 反向传播 求梯度
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])

    for t in reversed(xrange(len(inputs))):
        dy = np.copy(ps[t])  # copy ps[t] 的概率值
        dy[targets[t]] -= 1  # softmax 求导特点，对应位置减1，其他位置不变
        dWhy += np.dot(dy, hs[t].T)  # dWhy 是每一个Why导数的和
        dby += dy

        dh = np.dot(Why.T, dy) + dhnext  # dy(t)/dh(t) dy(t+1)/dh(t)
        dhraw = (1 - hs[t] * hs[t]) * dh  # back through thanh f(z) = tanh(z)  f(z)' = 1 − (f(z))^2
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t - 1].T)
        dhnext = np.dot(Whh.T, dhraw)

    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)  # 限制更新的参数不能过大

    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]


def sample(h, seed_ix, n):
    """
    采样
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for first time step
    """
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in xrange(n):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)  # (h,1)
        y = np.dot(Why, h) + by  # (v,1)
        p = np.exp(y) / np.sum(np.exp(y))  # (v,1)
        ix = np.random.choice(range(vocab_size), p=p.ravel())  # 将我维数组降为一维
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes


n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)  # memory variables for Adagrad
smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # loss at iteration 0
while True:
    # prepare inputs (we're sweeping from left to right in steps seq_length long)
    if p + seq_length + 1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size, 1))  # reset RNN memory
        p = 0  # go from start of data
    inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]] # 从data中任意取出一段
    targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]] # 找到对应inputs 每一个字母对应的下一个字
    # seq_length  25

    # if n % 100 == 0:
    #     print 'inputs:', data[p:p + seq_length]
    #     print 'targets:', targets

    # sample from the model now and then
    if n % 100 == 0:
        sample_ix = sample(hprev, inputs[0], 400)
        # len(sample_ix)  #200
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print '----\n %s \n----' % (txt,)

    # forward seq_length characters through the net and fetch gradient
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)

    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss)  # print progress

    # perform parameter update with Adagrad 更新参数
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                  [dWxh, dWhh, dWhy, dbh, dby],
                                  [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

    p += seq_length  # move data pointer  去截取下一段文本
    n += 1  # iteration counter
