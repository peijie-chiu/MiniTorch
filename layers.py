import numpy as np
import autograd

# Init Weights
def xavier(shape):
    sq = np.sqrt(3.0/np.prod(shape[:-1]))
    return np.random.uniform(-sq,sq,shape)


##################### Containers ################################
## This is still how pytorch formulates Sequential() container ## 
## We can add any layers to the container as they are callable ##
#################################################################
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

    def add_layer(self, layer):
        self.layers.append(layer)
        self.names.append(layer.__class__.__name__)


##################### Layers #############################
## for any new layers, we should rewrite __init__,      ##
## __call__, __str__ function to ensure the consistency ##
##########################################################
class Linear():
    def __init__(self, in_features, out_features):
        """
        Fully-Connected Layer

        Parameters:
        in_features: input feature size
        out_features: output feature size
        """
        self.in_features = in_features
        self.out_features = out_features
        self.w = autograd.Param()
        self.b = autograd.Param()
        
        self.w.set(xavier((in_features, out_features)))
        self.b.set(np.zeros((out_features)))           

    def __call__(self, x):
        x = autograd.matmul(x, self.w)
        x = autograd.add(x, self.b)

        return x

    def __str__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features})"


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=True):
        """
        Multi-channel 2D Convolutional Layer
        Performing valid 2D convolution

        Parameters:
        in_channels: number of input channels
        out_channels: number of filters
        kernel_size: the convolutional kernel size
        stride: controls the stride of the convolution
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias

        self.k = autograd.Param()
        self.k.set(xavier((kernel_size, kernel_size, in_channels, out_channels)))
        
        if bias:
            self.b = autograd.Param()
            self.b.set(np.zeros((self.out_channels)))

    def __call__(self, x):
        x = autograd.conv2(x, self.k, self.stride)
        if self.bias:
            x = autograd.add(x, self.b) 

        return x

    def __str__(self):
        return f"Conv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, bias={self.bias})"


class RELU():
    def __init__(self):
        pass

    def __call__(self, x):
        return autograd.RELU(x)

    def __str__(self):
        return f"RELU()"


class Flatten():
    def __init__(self):
        pass

    def __call__(self, x):
        return autograd.flatten(x)

    def __str__(self):
        return f"Flatten()"


class Down():
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return autograd.down(x, self.factor)

    def __str__(self):
        return f"Down(factor={self.factor})"



# Cross Entropy of Soft-max. 
# This is how CrossEntropyLoss in pytorch is implemented
class SmaxCELoss():
    def __init__(self):
        pass

    def __call__(self, y_pred, y_true):
        y = autograd.smaxloss(y_pred, y_true)
        y = autograd.mean(y)

        return y


# Accuracy
class Accuracy():
    def __init__(self):
        pass

    def __call__(self, y_pred, y_true):
        acc = autograd.accuracy(y_pred, y_true)
        acc = autograd.mean(acc)

        return acc

# main = Sequential([Conv2d(1,2,3, 1),
#                    RELU(),
#                    Linear(400, 10)
#                 ])

# print(main)