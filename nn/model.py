from nn import layer
from nn.container import Module, Sequential

class ConvNet(Module):
    def __init__(self, nc, nf, n_class, im_size):
        super().__init__()
        self.main = Sequential(name='main')
        channels = [nc] + nf
        for i in range(len(channels)-1):
            if i < len(channels)-1:
                self.main.add_module(layer.Conv2d(channels[i], channels[i+1], kernel_size=3, stride=1, padding=1))
                self.main.add_module(layer.BatchNorm2d(channels[i+1]))
                self.main.add_module(layer.RELU())
                self.main.add_module(layer.Maxpool2d(2, 2))
                
        # self.main.add_module(layer.Conv2d(channels[-1] , n_class, kernel_size=im_size // (2**len(nf)), stride=1, padding=0))
        self.main.add_module(layer.Flatten())
        self.main.add_module(layer.Dropout(0.3))
        self.main.add_module(layer.Linear((im_size // (2**len(nf)))**2*channels[-1], n_class))

    def forward(self, x):
        return self.main(x)
    
    def __str__(self):
        return self.main.__str__()


class MLP(Module):
    def __init__(self, nc, nf, n_class):
        super().__init__()
        self.main = Sequential()
        channels = [nc] + nf + [n_class]
        
        for i in range(len(channels)-1):
            self.main.add_module(layer.Linear(channels[i], channels[i+1]))
            if not i == len(channels)-2:
                self.main.add_module(layer.RELU())

    def forward(self, x):
        return self.main(x)

    def __str__(self):
        return self.main.__str__()