##################### Containers ################################
## This is still how pytorch formulates containers             ## 
## We can add any layers to the container as they are callable ##
#################################################################

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
        self.layers = []
        
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