%matplotlib inline
# display figures in the notebook
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

digits = load_digits()
sample_index = 45
plt.figure(figsize=(3, 3)), plt.imshow(digits.images[sample_index],cmap=plt.cm.gray_r,interpolation='nearest'), plt.title(
    "image label:%d" % digits.target[sample_index]);

from sklearn.model_selection import train_test_split


data = np.asarray(digits.data, dtype='float32')
target = np.asarray(digits.target, dtype='int32')

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.15, random_state=37)

data.shape
target.shape


from sklearn import preprocessing
# mean = 0 ; standard deviation = 1.0
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(scaler.mean_)
print(scaler.scale_)
