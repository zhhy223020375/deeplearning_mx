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