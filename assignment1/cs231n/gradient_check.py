# coding=UTF-8
import numpy as np
from random import randrange


def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
    a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at
    """

    fx = f(x)  # evaluate function value at original point
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        # evaluate function at x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h  # increment by h
        fxph = f(x)  # evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x)  # evaluate f(x - h)
        x[ix] = oldval  # restore

        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h)  # the slope
        if verbose:
            print ix, grad[ix]
        it.iternext()  # step to next dimension

    return grad


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


def eval_numerical_gradient_blobs(f, inputs, output, h=1e-5):
    """
    Compute numeric gradients for a function that operates on input
    and output blobs.

    We assume that f accepts several input blobs as arguments, followed by a blob
    into which outputs will be written. For example, f might be called like this:

    f(x, w, out)

    where x and w are input Blobs, and the result of f will be written to out.

    Inputs:
    - f: function
    - inputs: tuple of input blobs
    - output: output blob
    - h: step size
    """

    numeric_diffs = []
    for input_blob in inputs:
        diff = np.zeros_like(input_blob.diffs)
        it = np.nditer(input_blob.vals, flags=['multi_index'],
                       op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            orig = input_blob.vals[idx]

            input_blob.vals[idx] = orig + h
            f(*(inputs + (output,)))
            pos = np.copy(output.vals)
            input_blob.vals[idx] = orig - h
            f(*(inputs + (output,)))
            neg = np.copy(output.vals)
            input_blob.vals[idx] = orig

            diff[idx] = np.sum((pos - neg) * output.diffs) / (2.0 * h)

            it.iternext()
        numeric_diffs.append(diff)
    return numeric_diffs


def eval_numerical_gradient_net(net, inputs, output, h=1e-5):
    return eval_numerical_gradient_blobs(lambda *args: net.forward(),
                                         inputs, output, h=h)


def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
    """
    解析梯度
    sample a few random elements and only return numerical
    in this dimensions.
    """
    # grad_check_sparse(f, W, grad)
    # f = lambda w: svm_loss_naive(w, X_dev, y_dev, 0.0)[0]
    # svm_loss_naive(w, X_dev, y_dev, 0.0)[0]表示返回的Li,即损失函数的值 loss
    # svm_loss_naive(W, X, y, reg):
    # print 'analytic_grad is :::'
    # print analytic_grad
    for i in xrange(num_checks):
        # randrange() 方法返回指定递增基数集合中的一个随机数，基数缺省值为1。
        # W的维度 (3073L, 10L)
        ix = tuple([randrange(m) for m in x.shape])

        # num_checks=10，表示循环10次，每次随机出一个值

        # x[ix] 这里表示W中的对应的ix 的值， 获取其中一个w值，这里表示为oldval
        oldval = x[ix]
        x[ix] = oldval + h  # increment by h 把W中的一个值上调h
        fxph = f(x)  # 把新的w  带入f   evaluate f(x + h)

        x[ix] = oldval - h  # increment by h
        fxmh = f(x)  # evaluate f(x - h) 再把W中的一个值下调h
        x[ix] = oldval  # reset

        # 数字梯度
        grad_numerical = (fxph - fxmh) / (2 * h)# 这里就是求梯度的公式
        # 解析梯度
        grad_analytic = analytic_grad[ix]#这里的analytic_grad就是传入的grad
        # print 'grad_analytic change to '
        # print grad_analytic
        # 偏差

        rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
        print 'numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error)
