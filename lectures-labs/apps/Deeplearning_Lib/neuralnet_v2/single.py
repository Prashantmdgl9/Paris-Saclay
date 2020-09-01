import numpy as np
import sys
sys.path.append('/Users/prashantmudgal/Documents/Quantplex Labs/Paris-Saclay/lectures-labs/apps/Deeplearning_Lib/neuralnet_v2')
import loss
import data
import layers
import train

X, y = data.data()

OUTPUT_SIZE = 1

layer_obj = layers.Layer_Dense(OUTPUT_SIZE)

print(layer_obj.forward(X))


for i in range(1000):
    if (i % 100 == 0):
        print("Loss: " + str(np.mean(np.square(y - layer_obj.forward(X)))))
    train.train(X, y, layer_obj)


print("Input: " + str(X))
print("Actual Output: " + str(y))
print("Loss: " + str(np.mean(np.square(y - layer_obj.forward(X)))))
print("\n")
print("Predicted Output: " + str(layer_obj.forward(X)))
