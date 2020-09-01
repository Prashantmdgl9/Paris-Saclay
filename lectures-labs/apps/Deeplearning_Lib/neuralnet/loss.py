# A loss function measures how good our predictions
# Used to adjust params such as bias and weights

import numpy as np
import sys
sys.path.append('/Users/prashantmudgal/Documents/Quantplex Labs/Paris-Saclay/lectures-labs/apps/Deeplearning_Lib/')
from neuralnet.tensor import Tensor

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual:Tensor)-> Tensor:
        raise NotImplementedError

class MSE(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual)**2)
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual:Tensor)-> Tensor:
        return 2 * (predicted - actual)
