import numpy as np
import nn.autograd as autograd
from nn.container import Module

# Init Weights
def xavier(shape):
    sq = np.sqrt(3.0/np.prod(shape[:-1]))
    return np.random.uniform(-sq,sq,shape)


# The Placeholder for Inputs
class Tensor(autograd.Value):
    def __init__(self):
        super().__init__()

    def set(self, value):
        return super().set(value) 


##################### Layers #############################
## for any new layers, we should rewrite __init__,      ##
## __call__, __str__ function to ensure the consistency ##
##########################################################
class Linear(Module):
    def __init__(self, in_features, out_features):
        """
        Fully-Connected Layer

        Parameters:
        in_features: input feature size
        out_features: output feature size
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w = autograd.Param()
        self.b = autograd.Param()
        
        self.w.set(xavier((in_features, out_features)))
        self.b.set(np.zeros((out_features)))           

    def forward(self, x):
        x = autograd.Matmul(x, self.w)
        x = autograd.Add(x, self.b)

        return x

    def __str__(self):
        return f"{self.name}(in_features={self.in_features}, out_features={self.out_features})"


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=True):
        """
        Multi-channel 2D Convolutional Layer
        Performing valid 2D convolution

        Parameters:
        in_channels: number of input channels
        out_channels: number of filters
        kernel_size: the convolutional kernel size
        stride: controls the stride of the convolution
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.padding = padding

        self.k = autograd.Param()
        self.k.set(xavier((kernel_size, kernel_size, in_channels, out_channels)))
        
        if bias:
            self.b = autograd.Param()
            self.b.set(np.zeros((self.out_channels)))

    def forward(self, x):
        x = autograd.Conv2d(x, self.k, self.stride, self.padding)
        if self.bias:
            x = autograd.Add(x, self.b) 

        return x

    def __str__(self):
        return f"{self.name}(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, bias={self.bias})"


class RELU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return autograd.RELU(x)

    def __str__(self):
        return f"{self.name}()"


class Flatten(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return autograd.Flatten(x)

    def __str__(self):
        return f"{self.name}()"


class Maxpool2d(Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        return autograd.Maxpool2d(x, self.kernel_size, self.stride)

    def __str__(self):
        return f"{self.name}(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


class Down(Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return autograd.Down(x, self.factor)

    def __str__(self):
        return f"{self.name}(factor={self.factor})"


class Dropout(Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        return autograd.Dropout(x, self.p)

    def __str__(self):
        return f"{self.name}(p={self.p})"


class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine=affine

        self.gamma = autograd.Param()
        self.beta = autograd.Param()
        self.gamma.set(np.ones((1,1,1,self.num_features)))
        self.beta.set(np.zeros((1,1,1,self.num_features)))

    def forward(self, x):
        # if len(x.top.shape) == 4:
        #     self.gamma.set(np.ones((1,1,1,self.num_features)))
        #     self.beta.set(np.zeros((1,1,1,self.num_features)))
        # else:
        #     self.gamma.set(np.ones((1,self.num_features)))
        #     self.beta.set(np.ones((1,self.num_features)))
            
        return autograd.BatchNorm2d(x, self.num_features, self.gamma, self.beta, self.eps, self.momentum, self.affine)

    def __str__(self):
        return f"{self.name}(num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine})"
