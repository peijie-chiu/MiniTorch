##################### Containers ################################
## This is still how pytorch formulates containers             ## 
## We can add any layers to the container as they are callable ##
#################################################################

# This is the basic class for all the modules on layer.py, loss.py
class Module():
    def __init__(self):
        pass
    
    def __call__(self, *params):
        return self.forward(*params)

    def forward(self, *params):
        pass


# Sequential Model
class Sequential():
    def __init__(self, *layers):
        self.names = []
        self.layers = []
        if layers is not None:
            self.layers.extend(*layers)
            self.names.extend([l for l in self.layers])

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            self.names.append(layer.__class__.__name__)

        return x

    def __str__(self):
        info = 'Sequential(\n'
        for i, n in enumerate(self.names):
            info += f"  ({i}) {n}\n" 
        info += ')'
        return info

    def add_module(self, layer):
        self.layers.append(layer)
        self.names.append(layer.__class__.__name__)