##################################################
# The optimizer used to optimize the parameters 
# when training the neural network
##################################################
import numpy as np


class Solver():
    def __init__(self, params):
        """
        The parent abstract class of Solvers

        Parameters:
        params: the parameters in the neural network that
        the optimizer will optimize
        """
        self.params = params

    def step(self):
        raise Exception("Not Implemented")


# SGD
class SGD(Solver):
    def __init__(self, params, lr):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            p.top += - self.lr * p.grad


# Momentum
class Momentum(Solver):
    def __init__(self, params, lr, mom=0.9):
        super().__init__(params)
        self.lr = lr
        self.mom = mom

        self.init_momentum()

    def step(self):
        for p in self.params:
            p.momentum = p.grad + self.mom * p.momentum
            p.top += - self.lr * p.momentum
    
    def init_momentum(self):
        for p in self.params:
            p.momentum = np.zeros_like(p.top)


# Nesterov
class Nesterov(Solver):
    def __init__(self, params, lr, mom=0.9):
        super().__init__(params)
        self.lr = lr
        self.mom = mom

        self.init_momentum()

    def step(self):
        for p in self.params:
            p.momentum = self.mom * p.momentum + p.grad # velocity update stays the same
            p.top += - self.lr * (p.grad + self.mom * p.momentum) 

    def init_momentum(self):
        for p in self.params:
            p.momentum = np.zeros_like(p.top)

# Adagrad
class Adagrad(Solver):
    def __init__(self, params, lr, eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.eps = eps
        self.init_grad_hist()

    def step(self):
        for p in self.params:
            p.grad_hist += p.grad ** 2
            p.top += -self.lr * p.grad / (np.sqrt(p.grad_hist) + self.eps)
 
    def init_grad_hist(self):
        for p in self.params:
            p.grad_hist = np.zeros_like(p.top)

# RMSprop
class RMSprop(Solver):
    def __init__(self, params, lr, decay_rate=0.9, eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.decay_rate = decay_rate
        self.eps = eps
        self.init_grad_hist()

    def step(self):
        for p in self.params:
            p.grad_hist = self.decay_rate * p.grad_hist + (1-self.decay_rate) * p.grad ** 2
            p.top += -self.lr * p.grad / (np.sqrt(p.grad_hist) + self.eps)

    def init_grad_hist(self):
        for p in self.params:
            p.grad_hist = np.zeros_like(p.top)

# Adam 
class Adam(Solver):
    def __init__(self, params, lr, eps=1e-8, betas=(0.9, 0.999)):
        super().__init__(params)
        self.lr = lr
        self.eps = eps
        self.betas = betas
        self.init_grad_hist()

    def step(self, t):
        # t is your iteration counter going from 1 to infinity
        for p in self.params:
            p.momentum = self.betas[0] * p.momentum + (1 - self.betas[0]) * p.grad
            mt = p.momentum / (1-self.betas[0]**t)
            p.velocity = self.betas[1] * p.velocity + (1 - self.betas[1]) * p.grad ** 2
            vt = p.velocity / (1-self.betas[1]**t)
            p.top += -self.lr * mt / (np.sqrt(vt) + self.eps)

    def init_grad_hist(self):
        for p in self.params:
            p.momentum = np.zeros_like(p.top)
            p.velocity = np.ones_like(p.top)