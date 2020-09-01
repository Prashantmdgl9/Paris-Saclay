import numpy as np
import sys
sys.path.append('/Users/prashantmudgal/Documents/Quantplex Labs/Paris-Saclay/lectures-labs/apps/Deeplearning_Lib/neuralnet_v2')
import loss

class Layer_Dense:
    def __init__(self, OUTPUT_SIZE):
        self.OUTPUT_SIZE = 1

    def weights(self, dim1, dim2):
        return np.random.randn(dim1, dim2)

    def bias(self, dim1):
        return np.zeros(( dim1, 1))

    def sigmoid(self, val, prime = False):
        if (prime == True):
            return val * (1 - val)
        return 1/(1 + np.exp(-val))

    def forward(self, inputs):
        n_layers = 2
        #print(inputs.shape) # 3*2 matrix
        self.weights_l = []
        self.weights_l.append(self.weights(inputs.shape[1], inputs.shape[0]))
        self.weights_l.append(self.weights(inputs.shape[0], self.OUTPUT_SIZE))

        self.o1 = np.dot(inputs, self.weights_l[0]) #+ self.bias(inputs.shape[0])
        self.o2 = self.sigmoid(self.o1)
        self.o3 = np.dot(self.o2, self.weights_l[1]) #+ self.bias(inputs.shape[0])
        self.o4 = self.sigmoid(self.o3)
        #print(self.o4)
        return self.o4

    def backward_propagation(self, X, y, output):
        self.error = loss.error(y, output)
        self.output_delta = self.error * self.sigmoid(output, prime=True)
        # error for the hidden layer
        self.z2_error = self.output_delta.dot(self.weights_l[1].T) #z2 error: how much our hidden layer weights contribute to output error
        # delta of the hidden layer
        self.z2_delta = self.z2_error * self.sigmoid(self.o1, prime=True) #applying derivative of sigmoid to z2 error
        k = X.T.dot(self.z2_delta)
        self.weights_l[0] = self.weights_l[0] + k  # adjusting first set (input -> hidden) weights
        k = self.o1.T.dot(self.output_delta)
        self.weights_l[1] = self.weights_l[1] + k # adjusting second set (hidden -> output) weights


class NeuralNetwork(object):
    def __init__(self):
        #parameters
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3

        #weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (3x2) weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer

    def feedForward(self, X):
        #forward propogation through the network
        self.z = np.dot(X, self.W1) #dot product of X (input) and first set of weights (3x2)
        self.z2 = self.sigmoid(self.z) #activation function
        self.z3 = np.dot(self.z2, self.W2) #dot product of hidden layer (z2) and second set of weights (3x1)
        output = self.sigmoid(self.z3)
        return output

    def sigmoid(self, s, deriv=False):
        if (deriv == True):
            return s * (1 - s)
        return 1/(1 + np.exp(-s))

    def backward(self, X, y, output):
        #backward propogate through the network
        self.output_error = y - output # error in output
        self.output_delta = self.output_error * self.sigmoid(output, deriv=True)

        self.z2_error = self.output_delta.dot(self.W2.T) #z2 error: how much our hidden layer weights contribute to output error
        self.z2_delta = self.z2_error * self.sigmoid(self.z2, deriv=True) #applying derivative of sigmoid to z2 error

        self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input -> hidden) weights
        self.W2 += self.z2.T.dot(self.output_delta) # adjusting second set (hidden -> output) weights
