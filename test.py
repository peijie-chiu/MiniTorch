import numpy as np

from nn import graph
from nn import solver
from nn.loss import SmaxCELoss, accuracy
from nn.graph import Graph, Seesion
from nn.container import Placeholder, Sequential
from nn.layer import Linear, RELU, Conv2d, Flatten

sess = Seesion()
sess.close()
sess.close()

# Graph().as_default()

# inp = Placeholder()
# lab = Placeholder()

# main = Sequential([Conv2d(1, 16, 3, 1, 1),
#                     RELU(),
#                     Conv2d(16, 32, 3, 1, 1),
#                     Flatten(),
#                     Linear(800, 10)])
# print(main)
# y = main(inp)

# criterion = SmaxCELoss()
# loss = criterion(y, lab)

# sess = Seesion()
# sess.eval()
# output = sess.run(loss, {inp: np.arange(50).reshape(-1, 5, 5, 1), lab:np.ones((2, ))})
# acc = accuracy(y.top, np.ones((2, )))
# print(acc)
# print(loss.top, y.top.shape)

# # optim = solver.Adam(graph._default_graph.variables, 1e-3, 0.9)
# # optim.step(1)
