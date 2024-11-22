import numpy as np
from mxnet import nd

x = nd.arange(0, 10, 1)
print(x)
print(x.shape)
print(x.size)
print(x.reshape([2, 5]))
print(x)
print(x.reshape([2, 5]).shape)
print(x.size)

y = nd.zeros(shape=10)
print(y)
print(y.shape)

x = nd.arange(0, 12, 1)
y = nd.ones(shape=[3,4])
x = x.reshape([3, 4])
print(x+y)
print(x * y)
print(x / y)
print(nd.dot(x, y.T))

from mxnet import autograd
x = nd.arange(0, 4, 1).reshape(shape=[4,1])
print(x)
x.attach_grad()
with autograd.record():
    y = nd.dot(x.T, x)
    y.backward()
print(x.grad)

x = nd.arange(0, 4, 1).reshape(shape=[4,1])
x.attach_grad()
print(autograd.is_training())
with autograd.record():
    y = 3 * nd.dot(x.T, x)
    y.backward()
    print(x.grad)
    print(autograd.is_training())

x.attach_grad()
with autograd.record():
    y = 3 * x[0] + 4 * x[0] * x[0] + 2 * x[1] + 2 * x[2] + 2 * x[3]
    y.backward()
print(x.grad)