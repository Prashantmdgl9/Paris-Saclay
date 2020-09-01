import numpy as np
embedding_size = 4
vocab_size = 10

embedding_matrix = np.arange(emdedding_size * vocab_size, dtype = 'float32')
embedding_matrix = embedding_matrix.reshape(vocab_size, emdedding_size)
embedding_matrix.shape
print(embedding_matrix)
print(embedding_matrix[3])

# Compute a one-hot encoding vector v, then compute a dot product with the embedding matrix
i = 3
def onehot_encode(dim, label):
    return np.eye(dim)[label]

onehot_i = onehot_encode(vocab_size, i)
print(onehot_i)

embedding_vector = np.dot(onehot_i, embedding_matrix)
print(embedding_vector)

from tensorflow.keras.layers import Embedding
embedding_layer = Embedding(output_dim = embedding_size, input_dim = vocab_size, weights = [embedding_matrix], input_length = 1, name = 'my_embedding')

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
x = Input(shape=[1], name = 'input')
embedding  = embedding_layer(x)
model = Model(inputs=x, outputs = embedding)

# The output of an embedding layer is then a 3-d tensor of
# shape (batch_size, sequence_length, embedding_size).

model.output_shape
model.get_weights()
model.summary()
labels_to_encode = np.array([[3]])
model.predict(labels_to_encode)


labels_to_encode = np.array([[3], [3], [0], [9]])
model.predict(labels_to_encode)

from tensorflow.keras.layers import Flatten
x = Input(shape=[1], name = 'input')
y = Flatten() (embedding_layer(x))
model2 = Model(inputs=x, outputs=y)
model2.output_shape
model2.predict(np.array([3]))
model2.summary()
model2.set_weights([np.ones(shape=(vocab_size, embedding_size))])
labels_to_encode = np.array([[3]])
model2.predict(labels_to_encode)
model.predict(labels_to_encode)

from tensorflow.keras.models import Sequential
model3 = Sequential()
model3.add(embedding_layer)
model3.add(Flatten())

labels_to_encode = np.array([[3]])
print(model3.predict(labels_to_encode))
























mode
