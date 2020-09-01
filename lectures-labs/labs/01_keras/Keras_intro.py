# An intro to Keras
# MNIST Digit data classification using tensorflow and keras layers

# The data has 1797 images.
# Digits.data has flattened arrays of shape (64,) they are reshaped to (8,8) to work on them

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
import pandas as pd
import time
import pickle
import seaborn as sns
sns.set()
import glob

digits = load_digits()
sample_index = 45
def plot_digit(sample_index):
    plt.figure(figsize=(3, 3))
    plt.imshow(digits.images[sample_index], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title("image label: %d" % digits.target[sample_index])
plot_digit(sample_index)

#Train test split of the datasets
from sklearn.model_selection import train_test_split
data = np.asarray(digits.data, dtype='float32')
target = np.asarray(digits.target, dtype='int32')
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.2, random_state = 123)

#Preprocess the data
#Normalisation means pushing values between 0 and 1
#Standardisation means the values will be x-mean/standard deviation
#Trees don't require any scaling but other algos such as regression require scaling because they have to reach the maxima or minima quickly or need to find the min/max euclidean
#distances.
#Standardisation performs better most of the times than Normalisation
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
scaler.scale_.shape

plt.figure(figsize=(3, 3))
plt.imshow(X_train[sample_index].reshape(8, 8),
           cmap=plt.cm.gray_r, interpolation='nearest')
plt.title("transformed sample\n(standardization)");
#Post Scaling one can retrieve the scaled objects
def transform_back(sample_index):
    plt.figure(figsize=(3, 3))
    plt.imshow(scaler.inverse_transform(X_train[sample_index]).reshape(8,8), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title("Retransformed image")

transform_back(sample_index)

X_train.shape, y_train.shape
X_test.shape, y_test.shape
#Preprocessing of the Target Data
y_train[:3]

# One hot encoding of the output data
from tensorflow.keras.utils import to_categorical
Y_train = to_categorical(y_train)
Y_train[:3]
# A few checks on the data
X_train.shape
len(digits.data)
len(digits.target)
len(digits.images)
digits.images[0].shape
digits.data[0].shape


# Feed forward neural network with Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers

input_dim = X_train.shape[1] #64 in this case
hidden_dim =100
output_dim = 10
learning_rate = 0.1

model = Sequential()
model.add(Dense(hidden_dim, input_dim = input_dim, activation='relu'))
model.add(Dense(hidden_dim, activation='relu'))
model.add(Dense(output_dim, activation='softmax'))
model.compile(optimizer = optimizers.Adam(), loss = 'categorical_crossentropy', metrics=['accuracy'] )
#model.compile(optimizer=optimizers.SGD(learning_rate=learning_rate, momentum = 0.9, nesterov=True), loss = 'categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, Y_train, validation_split=0.2, epochs=15, batch_size=32)


#Wrapping it in pandas dataframe for plotting
history_df = pd.DataFrame(history.history)
history_df["epoch"] = history.epoch
history_df
path = 'models/'
model_name = "Dense_Adam_def_"+ str(learning_rate) +"_"+ str(hidden_dim) + "_" + time.strftime("%m_%d_%H_%M")
model.save(path + model_name + ".h5")

pick_file_name = 'histories/' + model_name + '.pkl'
outfile = open(pick_file_name, 'wb')
pickle.dump(history.history , outfile)
outfile.close()

files = []
for file in glob.glob("histories/*.pkl"):
    files.append(file.rsplit('/')[1])
files
#Select below and run to see graphs properly
def plt_pickles():
    plt.figure(figsize=(40,12))
    for val in files:
        with open('histories/' + val , 'rb') as f:
            load_f = pickle.load(f)
            ax = plt.subplot(221)
            plt.plot(load_f["loss"],label ='loss_' + val,linestyle='--')
            plt.plot(load_f["val_loss"],label='val_loss_' + val)
            plt.legend(fontsize=20)
            ax.legend(bbox_to_anchor=(1.1, 1.05))
            plt.ylabel('Loss')
            ax2 = plt.subplot(223)
            plt.plot(load_f["accuracy"],label ='accuracy' + val,linestyle='--')
            plt.plot(load_f["val_accuracy"],label='val_accuracy' + val)
            plt.legend(fontsize=20)
            ax2.legend(bbox_to_anchor=(1.1, 1.05))
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.subplot(221)
            plt.subplot(223)

plt_pickles()
# From above, reducing learning_rate decreases accuracy and increases the loss, it is because te learning rate is slow and thus in 15 epochs the desired values aren't reached

#Forward Pass and Generalisation
preds = model.predict_classes(X_test)
print("test acc: %0.4f" % np.mean(preds == y_test))


# One can use Tesorboard for visualising the loss in various models
%load_ext tensorboard
import datetime
from tensorflow.keras.callbacks import TensorBoard

model = Sequential()
model.add(Dense(hidden_dim, input_dim = input_dim, activation='tanh'))
model.add(Dense(hidden_dim, activation='tanh'))
model.add(Dense(output_dim, activation='softmax'))
model.compile(optimizer = optimizers.SGD(learning_rate=0.1), loss = 'categorical_crossentropy', metrics=['accuracy'] )
timestamp = datetime.datetime.now().strftime("Y%m%d-%H%M%S")
log_dir = "tensorboard_logs/" + timestamp
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)
model.fit(x=X_train, y=Y_train, validation_split=0.2, epochs=15, callbacks=[tensorboard_callback])
# Run the commmand below from command line rather than atom
!tensorboard --logdir tensorboard_logs



# Analysis of the exercise


# Setting the learning rate value to a small value (e.g. lr=0.001 on
# this dataset) makes the model train much slower (it has not
# converged yet after 15 epochs).
#
# Using momentum tends to mitigate the small learning rate / slow
# training problem a bit.
#
# Setting the learning rate to a very large value (e.g. lr=10)
# makes the model randomly bounce around a good local
# minimum and therefore prevent it to reach a low training loss even
# after 30 epochs.


# Adam with its default global learning rate of 0.001 tends to work
# in many settings often converge as fast or faster than SGD
# with a well tuned learning rate.
# Adam adapts the learning rate locally for each neuron, this is why
# tuning its default global learning rate is rarely needed.



# numpy arrays vs tensorflow tensors

predictions_tf = model(X_test)
predictions_tf[:5]
type(predictions_tf), predictions_tf.shape

import tensorflow as tf

tf.reduce_sum(predictions_tf, axis=1)[:5]

predicted_labels_tf = tf.argmax(predictions_tf, axis=1)
predicted_labels_tf[:5]

# We can compare those labels to the expected labels to compute the accuracy with the Tensorflow API.
# Note however that we need an explicit cast from boolean to floating point values to be able to compute
# the mean accuracy when using the tensorflow tensors:
accuracy_tf = tf.reduce_mean(tf.cast(predicted_labels_tf == y_test, tf.float64))
accuracy_tf

# TF tensors can be converted to numpy as well
accuracy_tf.numpy()
(predicted_labels_tf.numpy() == y_test).mean()


#Imapct of initialisation
# Keras' weights are initialised using Glorot Uniform initialization strategy
# each weight coefficient is randomly sampled from [-scale, scale]
#scale is proportional to $\frac{1}{\sqrt{n_{in} + n_{out}}}$

from tensorflow.keras import initializers
normal_init = initializers.TruncatedNormal(stddev=0.01)
model = Sequential()
model.add(Dense(hidden_dim, input_dim = X_train.shape[1], activation='tanh', kernel_initializer=normal_init))
model.add(Dense(hidden_dim, activation="tanh", kernel_initializer=normal_init))
model.add(Dense(output_dim, activation="softmax", kernel_initializer=normal_init))

model.compile(optimizer=optimizers.SGD(lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])
model.layers
model.layers[0].weights

w = model.layers[0].weights[0].numpy()
w
w.std()
b = model.layers[0].weights[1].numpy()
b #biases are 0 right now

history = model.fit(X_train, Y_train, epochs=15, batch_size=32)

plt.figure(figsize=(12, 4))
plt.plot(history.history['loss'], label="Truncated Normal init")
plt.legend();

model.layers[0].weights

# Let's change the initialisation
large_scale_init = initializers.TruncatedNormal(stddev=1)
small_scale_init = initializers.TruncatedNormal(stddev=1e-3)


optimizer_list = [
    ('SGD', optimizers.SGD(lr=0.1)),
    ('Adam', optimizers.Adam()),
    ('SGD + Nesterov momentum', optimizers.SGD(
            lr=0.1, momentum=0.9, nesterov=True)),
]

init_list = [
    ('glorot uniform init', 'glorot_uniform', '-'),
    ('small init scale', small_scale_init, '-'),
    ('large init scale', large_scale_init, '-'),
    ('zero init', 'zero', '--'),
]


for optimizer_name, optimizer in optimizer_list:
    print("Fitting with:", optimizer_name)
    plt.figure(figsize=(12, 6))
    for init_name, init, linestyle in init_list:
        model = Sequential()
        model.add(Dense(hidden_dim, input_dim=input_dim, activation="tanh",
                        kernel_initializer=init))
        model.add(Dense(hidden_dim, activation="tanh",
                        kernel_initializer=init))
        model.add(Dense(output_dim, activation="softmax",
                        kernel_initializer=init))

        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy')

        history = model.fit(X_train, Y_train,
                            epochs=10, batch_size=32, verbose=0)
        plt.plot(history.history['loss'], linestyle=linestyle,
                 label=init_name)

    plt.xlabel('# epochs')
    plt.ylabel('Training loss')
    plt.ylim(0, 6)
    plt.legend(loc='best');
    plt.title('Impact of initialization on convergence with %s'
              % optimizer_name)


# Just remember that if you network fails to learn at all (the loss stays at its initial
# value):
# - ensure that the weights are properly initialized,
# - inspect the per-layer gradient norms to help identify the bad layer,
# - use Adam instead of SGD as your default go to initializer.
#
#
