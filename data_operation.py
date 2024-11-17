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