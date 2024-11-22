from mxnet import autograd, nd
import random
num_input = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_input))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)
print(features)
print(labels)

w = nd.random.normal(scale=10, shape=(num_input, 1))
b = nd.random.normal(scale=1, shape=(1, ))

for epoch in range(20):
    w.attach_grad()
    b.attach_grad()
    with autograd.record():
        y_pred = nd.dot(features, w) + b
        l = 0.5 * (labels -y_pred.reshape(labels.shape)) ** 2
        l.backward()
        w = w - 0.001 * w.grad
        b = b - 0.001 * b.grad
    print(w)
    print(b)
    print(epoch)
    print(l.sum())

real_w = [2, 3, 4, 5]
real_x = nd.random.normal(loc=0, scale=1, shape=[1000, 3])
real_y = real_w[0] * real_x[:, 0] ** 2 + real_w[1] * real_x[:, 1] ** 3 + real_w[2] * real_x[:, 2] + real_w[3]
real_y = real_y + nd.random.normal(loc=0, scale=0.01, shape=[1000, ])
print(real_y)
pred_w = nd.random.normal(loc=0, scale=1, shape=[4,])
for epoch in range(10):
    pred_w.attach_grad()
    with autograd.record():
        pred_y = pred_w[0] * real_x[:, 0] ** 2 + pred_w[1] * real_x[:, 1] ** 3 + pred_w[2] * real_x[:, 2] + pred_w[3]
        loss = nd.sum((real_y - pred_y)**2)
        loss.backward()
        pred_w = pred_w - pred_w.grad * 0.00001
    print(pred_w)
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet import init
from mxnet.gluon import loss as gloss
from mxnet.gluon import Trainer
real_x = nd.random.normal(loc=0, scale=1, shape=[1000,])
real_y = 3 * real_x + 5 + nd.random.normal(loc=0, scale=0.01, shape=[1000,])
print(real_y)
dataset = gdata.ArrayDataset(real_x, real_y)
data_iter = gdata.DataLoader(dataset, batch_size=10, shuffle=True)

net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(init.Normal(sigma=0.01))
loss = gloss.L2Loss()
trainer = Trainer(net.collect_params(), 'sgd', {'learning_rate':0.03})
for epoch in range(3):
    for x, y in data_iter:
        with autograd.record():
            l = loss(net(x), y)
        l.backward()
        trainer.step(10)
    print(net[0].weight.data())
    print(net[0].bias.data())
