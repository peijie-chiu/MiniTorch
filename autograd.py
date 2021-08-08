########################################################
# An Autograd Engine mimicing the Pytorch/Tensorflow to 
# do automatic differentiation
########################################################
import numpy as np

# Global list of different kinds of components
ops = []
params = []
values = []

# Global forward
def Forward():
    for c in ops: c.forward()


# Global backward    
def Backward(loss):
    for c in ops:
        c.grad = np.zeros_like(c.top)
    for c in params:
        c.grad = np.zeros_like(c.top)

    loss.grad = np.ones_like(loss.top)
    for c in ops[::-1]: 
        c.backward() 


# SGD
def SGD(lr):
    for p in params:
        p.top = p.top - lr * p.grad


def init_momentum():
    for p in params:
        p.momentum=np.zeros_like(p.top)


# Heavy-ball method
def momentum(lr,mom=0.9):
    for p in params:
        p.momentum=p.grad+mom*p.momentum
        p.top=p.top-lr*p.momentum


# Values (Inputs)
class Value:
    def __init__(self):
        values.append(self)

    def set(self,value):
        self.top = np.float32(value).copy()

# Parameters (Weights we want to learn)
class Param:
    def __init__(self):
        params.append(self)

    def set(self,value):
        self.top = np.float32(value).copy()


################## Operations ##################

# Add layer (x + y) where y is same shape as x or is 1-D
class add:
    def __init__(self,x,y):
        ops.append(self)
        self.x = x
        self.y = y

    def forward(self):
        self.top = self.x.top + self.y.top

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + self.grad

        if self.y in ops or self.y in params:
            if len(self.y.top.shape) < len(self.grad.shape):
                ygrad = np.sum(self.grad,axis=tuple(range(len(self.grad.shape)-1)))
            else:
                ygrad= self.grad
            self.y.grad = self.y.grad + ygrad


# Matrix multiply (fully-connected layer)
class matmul:
    def __init__(self,x,y):
        ops.append(self)
        self.x = x
        self.y = y

    def forward(self):
        self.top = np.matmul(self.x.top,self.y.top)

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + np.matmul(self.y.top,self.grad.T).T
        if self.y in ops or self.y in params:
            self.y.grad = self.y.grad + np.matmul(self.x.top.T,self.grad)


# Rectified Linear Unit Activation            
class RELU:
    def __init__(self,x):
        ops.append(self)
        self.x = x

    def forward(self):
        self.top = np.maximum(self.x.top,0)

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + self.grad * (self.top > 0)


# Reduce to mean
class mean:
    def __init__(self,x):
        ops.append(self)
        self.x = x

    def forward(self):
        self.top = np.mean(self.x.top)

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + self.grad*np.ones_like(self.x.top) / np.float32(np.prod(self.x.top.shape))


# Soft-max + Loss (per-row / training example)
class smaxloss:
    def __init__(self,pred,gt):
        ops.append(self)
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
        if self.x in ops or self.x in params:
            truey = np.int64(self.y.top)
            self.save[range(len(truey)),truey] = self.save[range(len(truey)),truey] - 1.
            self.x.grad = self.x.grad + np.expand_dims(self.grad,-1)*self.save


# Compute accuracy (for display, not differentiable)        
class accuracy:
    def __init__(self,pred,gt):
        ops.append(self)
        self.x = pred
        self.y = gt

    def forward(self):
        truey = np.int64(self.y.top)
        self.top = np.float32(np.argmax(self.x.top,axis=1)==truey)

    def backward(self):
        # There is no need to back-propagate accuracy
        pass


# Downsample by a factor  
class down:
    def __init__(self,x,factor):
        ops.append(self)
        self.x = x
        self.factor = factor
        
    def forward(self):
        self.top = self.x.top[:,::self.factor,::self.factor,:]

    def backward(self):
        if self.x in ops or self.x in params:
            grd = np.zeros_like(self.x.top)
            grd[:,::self.factor,::self.factor,:] = self.grad
            self.x.grad = self.x.grad + grd


# Flatten (conv to fc)
class flatten:
    def __init__(self,x):
        ops.append(self)
        self.x = x
        
    def forward(self):
        self.top = np.reshape(self.x.top,[self.x.top.shape[0],-1])

    def backward(self):
        if self.x in ops or self.x in params:
            self.x.grad = self.x.grad + np.reshape(self.grad,self.x.top.shape)


# Convolution Layer
class conv2:
    def __init__(self, x, k, s=1):
        """
        Parameters:
        x: a input tensor with size of B, H, W, C1
        k: a multi-channel convolutional kernel with size of KH, KW, C1, C2
        s: controls the stride of the convolution
        """
        ops.append(self)
        self.x = x
        self.k = k
        self.s = s

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
        _, self.H, self.W, self.C1 = self.x.top.shape
        self.KH, self.KW, _, self.C2 = self.k.top.shape
        self.H_out = (self.H - self.KH) // self.s + 1
        self.W_out = (self.W - self.KW) // self.s + 1

        i, j, m = self.im2col_indices()
    
        x_crop = self.x.top[:, i, j, m] # Bx(KHxKWxC_in)x(H_outxW_out)
        k_crop = self.k.top.reshape(-1, self.C2)
        self.top = x_crop.transpose(0,2,1).dot(k_crop)
        self.top = self.top.reshape(-1, self.H_out, self.W_out, self.C2)
        
    def backward(self):
         ygrad = self.grad.reshape([-1, self.C2])

         if self.x in ops or self.x in params: 
             xgrad = np.zeros_like(self.x.top)
             i, j, m = self.im2col_indices()
             kcrop = self.k.top.reshape(-1, self.C2)
             xcrop = kcrop.dot(ygrad.T)
             xcrop = xcrop.reshape(self.C1*self.KH*self.KW, self.H_out*self.W_out, -1).transpose(2,0,1)
             np.add.at(xgrad, (slice(None),i,j,m), xcrop)

             self.x.grad += xgrad
            
         if self.k in ops or self.k in params:
             i, j, m = self.im2col_indices()
             xcrop = self.x.top[:,i,j,m].transpose(1,2,0).reshape(self.KH * self.KW * self.C1, -1)
             kgrad = xcrop.dot(ygrad).T.reshape(self.k.top.shape)
             
             self.k.grad += kgrad