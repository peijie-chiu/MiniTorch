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
            p.top = p.top - self.lr * p.grad


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
            p.top = p.top - self.lr * p.momentum
    
    def init_momentum(self):
        for p in self.params:
            p.momentum = np.zeros_like(p.top)


# Nesterov
class Nesterov(Solver):
    def __init__(self, params):
        super().__init__(params)

    def step(self):
        return super().step()


# Adam 
class Adam(Solver):
    def __init__(self, params):
        super().__init__(params)

    def step(self):
        return super().step()


# RMSprop
class RMSprop(Solver):
    def __init__(self, params):
        super().__init__(params)

    def step(self):
        return super().step()


# Adadelta
class Adadelta(Solver):
    def __init__(self, params):
        super().__init__(params)

    def step(self):
        return super().step()