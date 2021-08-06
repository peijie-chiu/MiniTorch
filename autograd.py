########################################################
# An Autograd Engine mimicing the Pytorch/Tensorflow
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
    for c in ops[::-1]: c.backward() 


# SGD
def SGD(lr):
    for p in params:
        p.top = p.top - lr*p.grad


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
        y = y - np.log(yS); yE = yE / yS

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
        pass

# Downsample by 2    
class down2:
    def __init__(self,x):
        ops.append(self)
        self.x = x
        
    def forward(self):
        self.top = self.x.top[:,::2,::2,:]

    def backward(self):
        if self.x in ops or self.x in params:
            grd = np.zeros_like(self.x.top)
            grd[:,::2,::2,:] = self.grad
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
        k: a multi-channel convolutional kernel with size of K1, K2, C1, C2
        s: controls the stride of the convolution
        """
        ops.append(self)
        self.x = x
        self.k = k
        self.s = s

    def im2col_indices(self):
        _, H, W, C1 = self.x.top.shape
        KH, KW, _, _ = self.k.top.shape

        assert (H - KH) % self.s  == 0, 'height does not work'
        assert (W - KW) % self.s  == 0, 'width does not work'

        H_out = H - KH + 1
        W_out = W - KW + 1

        i0 = np.tile(np.repeat(np.arange(KH), KW), C1)
        i1 = self.s * np.repeat(np.arange(H_out), W_out)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)

        j0 = np.tile(np.arange(KW), KH * C1)
        j1 = self.s * np.tile(np.arange(W_out), H_out)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        m = np.repeat(np.arange(C1), KH * KW).reshape(-1, 1)

        return i, j, m

    def forward(self):
        _, H, W, _ = self.x.top.shape
        KH, KW, _, C2 = self.k.top.shape

        H_out = H - KH + 1
        W_out = W - KW + 1

        i, j, m = self.im2col_indices()
    
        # xcrop = self.x.top.transpose(0,3,1,2)[:, m, i, j]
        # kcrop = self.k.top.transpose(3,2,0,1).reshape(C2, -1)

        # self.top = kcrop.dot(xcrop).transpose(1, 0, 2)
        # self.top = self.top.reshape([-1, C2, H_out, W_out]).transpose(0, 2, 3, 1)

        x_crop = self.x.top[:, i, j, m]
        k_crop = self.k.top.reshape(-1, C2)
        self.top = k_crop.T.dot(x_crop)
        # print(top.shape)
        self.top = self.top.transpose(0,2,1).reshape(-1, H_out, W_out, C2)
        
    def backward(self):
         if self.x in ops or self.x in params: 
             B, H, W, C1 = self.x.grad.shape
             K1, K2, C1, C2 = self.k.grad.shape

             x_grad = np.zeros_like(self.x.grad)
             k_grad = np.zeros_like(self.k.grad)
        
         if self.k in ops or self.k in params:
             B, H, W, C1 = self.x.top.shape
             K1, K2, C1, C2 = self.k.grad.shape

# x = Value()
# x.set(np.arange(1, 97, 1).reshape([2, 4, 4, 3]))
# k = Param()
# k.set(np.arange(1, 25, 1).reshape([2, 2, 3, 2]))

# conv = conv2(x, k)
# conv.forward()