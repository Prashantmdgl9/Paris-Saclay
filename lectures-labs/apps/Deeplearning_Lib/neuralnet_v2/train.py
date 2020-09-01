import numpy as np
import sys
sys.path.append('/Users/prashantmudgal/Documents/Quantplex Labs/Paris-Saclay/lectures-labs/apps/Deeplearning_Lib/neuralnet_v2')
import layers

def train(X, y, layers_obj):
    output = layers_obj.forward(X)
    layers_obj.backward_propagation(X, y, output)
