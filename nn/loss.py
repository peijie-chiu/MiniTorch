import nn.autograd as autograd
from nn.container import Module


class Regularization(Module):
    def __init__(self, lmd, type='l2'):
        super().__init__()
        self.lmd = lmd
        self.type = type

    def forward(self):
        pass

    def backward(self):
        pass


# Cross Entropy of Soft-max. 
# This is how CrossEntropyLoss in pytorch is implemented
class SmaxCELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        y = autograd.Smaxloss(y_pred, y_true)
        y = autograd.Mean(y)

        return y


# Accuracy
class Accuracy(Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        acc = autograd.Accuracy(y_pred, y_true)
        acc = autograd.Mean(acc)

        return acc