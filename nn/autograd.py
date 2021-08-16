########################################################
# An Autograd Engine mimicing the Pytorch/Tensorflow to 
# do automatic differentiation
########################################################
import numpy as np
from nn import graph
from nn.container import Module

################## Operations ##################
class Operation(Module):
    def __init__(self):
        if self not in graph._default_graph.ops:
            graph._default_graph.ops.append(self)

    def forward(self):
        raise Exception("Not Implemented")

    def backward(self):
        raise Exception("Not Implemented")


# Add layer (x + y) where y is same shape as x or is 1-D
class Add(Operation):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def forward(self):
        self.top = self.x.top + self.y.top

    def backward(self):
        if self.x in graph._default_graph.ops or self.x in graph._default_graph.variables:
            self.x.grad = self.x.grad + self.grad

        if self.y in graph._default_graph.ops or self.y in graph._default_graph.variables:
            if len(self.y.top.shape) < len(self.grad.shape):
                ygrad = np.sum(self.grad,axis=tuple(range(len(self.grad.shape)-1)))
            else:
                ygrad= self.grad
            self.y.grad = self.y.grad + ygrad


# Matrix multiply (fully-connected layer)
class Matmul(Operation):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def forward(self):
        self.top = np.matmul(self.x.top,self.y.top)

    def backward(self):
        if self.x in graph._default_graph.ops or self.x in graph._default_graph.variables:
            self.x.grad = self.x.grad + np.matmul(self.y.top,self.grad.T).T
        if self.y in graph._default_graph.ops or self.y in graph._default_graph.variables:
            self.y.grad = self.y.grad + np.matmul(self.x.top.T,self.grad)


# Rectified Linear Unit Activation            
class RELU(Operation):
    def __init__(self,x):
        super().__init__()
        self.x = x

    def forward(self):
        self.top = np.maximum(self.x.top,0)

    def backward(self):
        if self.x in graph._default_graph.ops or self.x in graph._default_graph.variables:
            self.x.grad = self.x.grad + self.grad * (self.top > 0)


# Reduce to mean
class Mean(Operation):
    def __init__(self,x):
        super().__init__()
        self.x = x

    def forward(self):
        self.top = np.mean(self.x.top)

    def backward(self):
        if self.x in graph._default_graph.ops or self.x in graph._default_graph.variables:
            self.x.grad = self.x.grad + self.grad*np.ones_like(self.x.top) / np.float32(np.prod(self.x.top.shape))


# Soft-max + Loss (per-row / training example)
class Smaxloss(Operation):
    def __init__(self,pred,gt):
        super().__init__()
        self.x = pred
        self.y = gt

    def forward(self):
        y = self.x.top
        y = y - np.amax(y,axis=1,keepdims=True)
        yE = np.exp(y)
        yS = np.sum(yE,axis=1,keepdims=True)
        y = y - np.log(yS)
        yE = yE / yS

        truey = np.int64(self.y.top)
        self.top = -y[range(len(truey)),truey]
        self.save = yE

    def backward(self):
        if self.x in graph._default_graph.ops or self.x in graph._default_graph.variables:
            truey = np.int64(self.y.top)
            self.save[range(len(truey)),truey] = self.save[range(len(truey)),truey] - 1.
            self.x.grad = self.x.grad + np.expand_dims(self.grad,-1)*self.save


# Compute accuracy (for display, not differentiable)        
class Accuracy(Operation):
    def __init__(self,pred,gt):
        super().__init__()
        self.x = pred
        self.y = gt

    def forward(self):
        truey = np.int64(self.y.top)
        self.top = np.float32(np.argmax(self.x.top,axis=1)==truey)

    def backward(self):
        # There is no need to back-propagate accuracy
        pass


# Downsample by a factor  
class Down(Operation):
    def __init__(self,x,factor):
        super().__init__()
        self.x = x
        self.factor = factor
        
    def forward(self):
        self.top = self.x.top[:,::self.factor,::self.factor,:]

    def backward(self):
        if self.x in graph._default_graph.ops or self.x in graph._default_graph.variables:
            grd = np.zeros_like(self.x.top)
            grd[:,::self.factor,::self.factor,:] = self.grad
            self.x.grad = self.x.grad + grd


# Flatten (conv to fc)
class Flatten(Operation):
    def __init__(self, x):
        super().__init__()
        self.x = x
        
    def forward(self):
        self.top = np.reshape(self.x.top,[self.x.top.shape[0],-1])

    def backward(self):
        if self.x in graph._default_graph.ops or self.x in graph._default_graph.variables:
            self.x.grad = self.x.grad + np.reshape(self.grad,self.x.top.shape)


# dropout layer
class Dropout(Operation):
    def __init__(self, x, p=0.5):
        super().__init__()
        self.x = x
        self.p = p
        
    def forward(self):
        if self.training:
            # Add dropout layer only in training phase
            self.r = np.random.binomial(1, self.p, size=self.x.top.shape) / self.p
            self.top = self.x.top * self.r
        else:
            # If in evaluation, the dropout layer do nothing
            self.top = self.x.top

    def backward(self):
        if self.x in graph._default_graph.ops or self.x in graph._default_graph.variables:
            self.x.grad = self.x.grad + self.grad * self.r


# 2d Maxpooling layer 
class Maxpool2d(Operation):
    def __init__(self, x, kernel_size=2, stride=2):
        super().__init__()
        self.x = x
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self):
        B, H, W, C = self.x.top.shape
        same_size = self.kernel_size == self.stride
        tiles = H % self.kernel_size == 0 and W % self.kernel_size  == 0

        assert same_size and tiles, "Please padding your input so that they have even dimension"
        
        self.x_reshaped = self.x.top.reshape(B, H // self.kernel_size, self.kernel_size,
                         W // self.kernel_size, self.kernel_size, C) 
        self.top = self.x_reshaped.max(axis=2).max(axis=3)
    
    def backward(self):
        if self.x in graph._default_graph.ops or self.x in graph._default_graph.variables:
            xgrad_reshaped = np.zeros_like(self.x_reshaped)
            out_newaxis = self.top[:, :, np.newaxis, :, np.newaxis, :]
            mask = (self.x_reshaped == out_newaxis)
            dout_newaxis = self.grad[:, :, np.newaxis, :, np.newaxis, :]
            dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, xgrad_reshaped)
            xgrad_reshaped[mask] = dout_broadcast[mask]
            xgrad_reshaped /= np.sum(mask, axis=(2, 4), keepdims=True)
            xgrad = xgrad_reshaped.reshape(self.x.top.shape)

            self.x.grad += xgrad


# 2d bactch normalization layer
class BatchNorm2d(Operation):
    def __init__(self, x, gamma, beta, eps=1e-5, momentum=0.1):
        super().__init__()
        self.x = x
        self.gamma = gamma
        self.beta = beta
        self.eps = eps
        self.momentum = momentum

        self.moving_mean = np.zeros((1,1,1,self.gamma.shape[-1]))
        self.moving_var = np.ones((1,1,1,self.gamma.shape[-1]))

    def forward(self):
        assert len(self.x.top.shape) == 4, 'The dimension must be BxHxWxC'

        if self.training:
            self.moving_mean = np.zeros((1,1,1,self.num_features))
            self.moving_var = np.ones((1,1,1,self.num_features))
            self.mu = self.x.top.mean(axis=(0,1,2), keepdims=True)
            self.var = self.x.top.var(axis=(0,1,2), keepdims=True)
        
            self.x_norm = (self.x.top - self.mu) / np.sqrt(self.var + self.eps)
            self.top = self.gamma.top * self.x_norm + self.beta.top

            self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * self.mu
            self.moving_var = self.momentum * self.moving_var + (1 - self.momentum) * self.var
        else:
            x_norm = (self.x.top - self.moving_mean) / np.sqrt(self.moving_var + self.eps)
            self.top = self.gamma.top * x_norm + self.beta.top

    def backward(self):
        B, H, W, _ = self.x.top.shape
        x_mu = self.x.top - self.mu
        std_inv = 1. / np.sqrt(self.var + self.eps)
        dx_norm = self.grad * self.gamma.top
        dvar = np.sum(dx_norm * x_mu, axis=(0,1,2), keepdims=True) * -.5 * std_inv**3
        dmu = np.sum(dx_norm * -std_inv, axis=(0,1,2), keepdims=True) + dvar * np.mean(-2. * x_mu, axis=(0,1,2), keepdims=True)

        if self.x in graph._default_graph.ops or self.x in graph._default_graph.variables:
             xgrad = (dx_norm * std_inv) + (dvar * 2 * x_mu / (B*W*H)) + (dmu / (B*W*H))
             self.x.grad += xgrad 

        if self.gamma in graph._default_graph.ops or self.gamma in graph._default_graph.variables:
            gamma_grad = np.sum(self.grad * self.x_norm, axis=(0,1,2), keepdims=True)
            self.gamma.grad += gamma_grad


        if self.beta in graph._default_graph.ops or self.beta in graph._default_graph.variables:
            beta_grad = np.sum(self.grad, axis=(0,1,2), keepdims=True) 
            self.beta.grad += beta_grad


# 2d instance normalization layer
class InstanceNorm2d(Operation):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()

    def forward(self):
        pass

    def backward(self):
        pass


# 2d Convolution Layer
class Conv2d(Operation):
    def __init__(self, x, k, s=1, pad=0):
        """
        Parameters:
        x: a input tensor with size of B, H, W, C1
        k: a multi-channel convolutional kernel with size of KH, KW, C1, C2
        s: controls the stride of the convolution
        """
        super().__init__()
        self.x = x
        self.k = k
        self.s = s
        self.pad = pad

    def im2col_indices(self):
        # assert (self.H - self.KH) % self.s  == 0, 'height does not work'
        # assert (self.W - self.KW) % self.s  == 0, 'width does not work'

        i0 = np.tile(np.repeat(np.arange(self.KH), self.KW), self.C1)
        i1 = self.s * np.repeat(np.arange(self.H_out), self.W_out)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)

        j0 = np.tile(np.arange(self.KW), self.KH * self.C1)
        j1 = self.s * np.tile(np.arange(self.W_out), self.H_out)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        m = np.repeat(np.arange(self.C1), self.KH * self.KW).reshape(-1, 1)

        return i, j, m

    def forward(self):
        self.B, self.H, self.W, self.C1 = self.x.top.shape
        self.KH, self.KW, _, self.C2 = self.k.top.shape
        self.H_out = (self.H - self.KH + 2 * self.pad) // self.s + 1
        self.W_out = (self.W - self.KW + 2 * self.pad) // self.s + 1

        i, j, m = self.im2col_indices()
        x_padded = np.pad(self.x.top, ((0, 0), (self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode='constant')
        x_crop = x_padded[:, i, j, m] # Bx(KHxKWxC_in)x(H_outxW_out)
        k_crop = self.k.top.reshape(-1, self.C2)
        self.top = x_crop.transpose(0,2,1).dot(k_crop)
        self.top = self.top.reshape(-1, self.H_out, self.W_out, self.C2)
        
    def backward(self):
         ygrad = self.grad.reshape([-1, self.C2])

         if self.x in graph._default_graph.ops or self.x in graph._default_graph.variables: 
             H_padded, W_padded = self.H + 2 * self.pad, self.W + 2 * self.pad
             xgrad = np.zeros((self.B, H_padded, W_padded, self.C1))
             i, j, m = self.im2col_indices()
             kcrop = self.k.top.reshape(-1, self.C2)
             xcrop = kcrop.dot(ygrad.T)
             xcrop = xcrop.reshape(self.C1*self.KH*self.KW, -1, self.B).transpose(2,0,1)
             np.add.at(xgrad, (slice(None),i,j,m), xcrop)

             if not self.pad == 0:
                xgrad = xgrad[:, self.pad:-self.pad, self.pad:-self.pad, :]

             self.x.grad += xgrad
            
         if self.k in graph._default_graph.ops or self.k in graph._default_graph.variables:
             i, j, m = self.im2col_indices()
             x_padded = np.pad(self.x.top, ((0, 0), (self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode='constant')
             xcrop = x_padded[:,i,j,m].transpose(1,2,0).reshape(self.KH * self.KW * self.C1, -1)
             kgrad = xcrop.dot(ygrad).T.reshape(self.k.top.shape)
             
             self.k.grad += kgrad
