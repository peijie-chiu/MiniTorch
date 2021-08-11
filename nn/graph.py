import numpy as np

# Global list of different kinds of components
ops = []
params = []
values = []

def train():
    for c in ops: c.train()

def eval():
    for c in ops: c.eval()

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