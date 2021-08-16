##################### Containers ################################
## This is still how pytorch formulates containers             ## 
## We can add any layers to the container as they are callable ##
#################################################################
import numpy as np
from nn import graph

# The Placeholder for Inputs
class Placeholder():
    def __init__(self):
        graph._default_graph.placeholders.append(self)

    def set(self,value):
        self.top = np.float32(value).copy()


# Parameters (Weights we want to learn)
class Variable():
    def __init__(self, value=None):
        graph._default_graph.variables.append(self)

        if value is not None:
            self.top = np.float32(value).copy()

    def set(self,value):
        self.top = np.float32(value).copy()


# This is the basic class for all the modules on layer.py, loss.py
class Module():
    def __init__(self):
        self.name = self.__class__.__name__
    
    def __call__(self, *params):
        return self.forward(*params)

    def forward(self, *params):
        raise Exception("Not Implemented")

    def backward(self, *params):
        raise Exception("Not Implemented")
    
    def train(self, mode=True):
        self.training = mode

    def eval(self):
        return self.train(False)


# Sequential Model
class Sequential(Module):
    def __init__(self, *modules, name='main'):
        super().__init__()
        self.name = name
        self.modules = []
        
        if modules:
            self.modules.extend(*modules)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        for m in self.modules:
            x = m(x)

        return x
    
    def __str__(self):
        info = f'({self.name}) Sequential(\n'
        for i, m in enumerate(self.modules):
            info += f"  ({i}) {m}\n" 
        info += ')'
        return info

    def add_module(self, module):
        self.modules.append(module)