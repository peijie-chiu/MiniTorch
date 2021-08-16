import numpy as np

class Graph():
    def __init__(self):
        # Global list of different kinds of components
        self.ops = []
        self.variables = []
        self.placeholders = []

    def as_default(self):
        global _default_graph
        _default_graph = self


class Seesion():
    def __init__(self):
        self.training = True

    def train(self):
        for c in _default_graph.ops: c.train()
        self.training = True

    def eval(self):
        for c in _default_graph.ops: c.eval()
        self.training = False
    
    def run(self, operation, feed_dict={}):
        for placeholder in _default_graph.placeholders:
            if placeholder in feed_dict.keys():
                placeholder.set(feed_dict[placeholder])

        for c in _default_graph.ops:  
            c.forward()
    
        if self.training:
            for c in _default_graph.ops:
                c.grad = np.zeros_like(c.top)
            for c in _default_graph.variables:
                c.grad = np.zeros_like(c.top)

            operation.grad = np.ones_like(operation.top)
            for c in _default_graph.ops[::-1]:
                c.backward() 

        return operation.top

    def __del__(self):
        print("Session Closed")