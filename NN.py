import numpy as np
import random as rnd


# Functions
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def tanh(x):
    return np.tanh(x)


# Cost functions and their deltas
class CrossEntropy:
    @staticmethod
    def cost(a, y):
        return np.sum(y * np.nan_to_num(np.log(a)) - (1 - y) * np.nan_to_num(np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        return a - y


class QuadraticCost:
    @staticmethod
    def cost(a, y):
        return np.linalg.norm(a - y)

    @staticmethod
    def delta(z, a, y):
        return (a - y) * sigmoid_prime(z)


class LogLikelihood:
    @staticmethod
    def cost(a, y):
        return -np.log(a[y])

    @staticmethod
    def delta(z, a, y):
        return a - y


# Neural network class
class Network:
    def __init__(self, layers, cost=CrossEntropy, transfer_fns=None):
        num_layers = len(layers)

        self.layers = layers
        self.num_layers = num_layers
        self.cost = cost

        if transfer_fns is None:
            self.transfer_fns = [sigmoid] * (num_layers - 1)
        else:
            self.transfer_fns = transfer_fns

        self.W = [(np.random.randn(i, j) / np.sqrt(j)) for i, j in zip(layers[1:], layers[:-1])]
        self.b = [np.random.randn(i, 1) for i in layers[1:]]

    def test_network(self, data):
        acc = 0

        for x, y in data:
            a, _ = self.feedforward(x)
            acc += (a[-1].argmax() == y)

        return acc

    def train_network(self, train_data, val_data, epochs, nbatch, eta, lmbda):
        num_samples = len(train_data)
        num_test = len(val_data)

        for e in range(epochs):
            rnd.shuffle(train_data)
            mini_batches = [train_data[ix:ix + nbatch] for ix in range(0, num_samples, nbatch)]

            for batch in mini_batches:
                self.update_mini_batch(batch, nbatch, eta, lmbda, num_samples)

            acc = self.test_network(val_data)
            print("Accuracy after training epoch {}: {} / {})".format(e, acc, num_test))

    def update_mini_batch(self, data, nbatch, eta, lmbda, N):
        W_delta = [np.zeros(W.shape) for W in self.W]
        b_delta = [np.zeros(b.shape) for b in self.b]

        decay = (eta * lmbda) / N
        for x, y in data:
            a, deltas = self.backpropagate(x, y)

            W_delta = [W_del + np.outer(delta, a_1) for (W_del, delta, a_1) in zip(W_delta, deltas, a[:-1])]
            b_delta = [b_del + delta for (b_del, delta) in zip(b_delta, deltas)]

        self.W = [(1 - decay) * W - (eta / nbatch) * W_del for (W, W_del) in zip(self.W, W_delta)]
        self.b = [b - (eta / nbatch) * b_del for (b, b_del) in zip(self.b, b_delta)]

    def backpropagate(self, x, y):
        delta = [None] * (self.num_layers - 1)
        a, z = self.feedforward(x)

        delta[-1] = self.cost.delta(z[-1], a[-1], y)

        for ix in range(2, self.num_layers):
            delta[-ix] = np.dot(np.transpose(self.W[-ix + 1]), delta[-ix + 1]) * sigmoid_prime(z[-ix])

        return a, delta

    def feedforward(self, x):
        a = [x]
        z = []

        for ix, (W, b) in enumerate(zip(self.W, self.b)):
            z.append(np.dot(W, a[-1]) + b)
            a.append(self.transfer_fns[ix](z[-1]))

        return a, z
