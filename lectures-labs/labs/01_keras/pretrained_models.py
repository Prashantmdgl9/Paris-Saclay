# In this code I will test ResNet50, MobileNet, and InceptionResNetV2 for their accuracy and ability to measure common items.
# These models are trained on imagenet

#Data Download Test
import sys
print('Python command used for this code')
print(sys.executable)
import tensorflow as tf
print('tensorflow:', tf.__version__)
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50

model = ResNet50(weights='imagenet')

from skimage.io import imread
import cv2
from skimage.transform import resize
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

image = imread('laptop.jpeg')
type(image)
image.shape
image.dtype
image.min()
image.max()
plt.imshow(image)
 np.product(image.shape)
image.nbytes
red_channel = image[:,:,0]
red_channel.min()
red_channel.max()
plt.imshow(red_channel)


plt.imshow(image[:,:,0], cmap=plt.cm.Reds_r)

green_channel = image[:,:,1]
blue_channel = image[:,:,2]
red_channel.shape
green_channel.shape
blue_channel.shape


# Mean can be calculated in numpy using np.mean rather than explicitly finding sum and then average of the matrices
grey_image = np.mean(image, axis=2)
print("Shape:{}".format(grey_image.shape))
print("Type: {}".format(grey_image.dtype))
print("image size: {:0.3} MB".format(grey_image.nbytes / 1e6))
print("Min: {}; Max: {}".format(grey_image.min(), grey_image.max()))

plt.imshow(grey_image, cmap=plt.cm.Greys_r)


#Resizing images, handling data types and dynamic ranges
from skimage.transform import resize

image = imread('laptop.jpeg')
image.shape

lowres_image = resize(image, (50, 50), mode = 'reflect', anti_aliasing=True)
lowres_image.shape
plt.imshow(lowres_image, interpolation='nearest')

lowres_image.dtype
print("image size:{} MB".format(lowres_image.nbytes/ 1e6))

lowres_image.min()
lowres_image.max()

#Keras requires images to be 0-255
low_Res_large_range = resize(image, (50,50), mode='reflect', anti_aliasing=True, preserve_range=True)
low_Res_large_range.shape
low_Res_large_range.min(), low_Res_large_range.max()
#The below won't work because there are float values in the 0-255 range
plt.imshow(low_Res_large_range, interpolation='nearest')

#Image Classification
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
model = ResNet50(weights='imagenet')

import tensorflow.keras.backend as K
K.image_data_format()
model.input_shape

#None in the above result - None, 224, 224, 3 means that batch size is None. Batch size = number of images that can be processed at one time
image = imread('laptop.jpeg')
image_224 = resize(image, (224, 224), preserve_range=True, mode = 'reflect')
image_224.shape
image_224.dtype
image_224 = image_224.astype(np.float32)
image_224.dtype


plt.imshow(image_224/255)
model.input_shape
image_224.shape
image_224_batch = np.expand_dims(image_224, axis = 0)
image_224_batch.shape


preds = model.predict(image_224_batch)
type(preds)
preds.dtype, preds.shape, preds.sum(axis=1)

from tensorflow.keras.applications.resnet50 import decode_predictions
decode_predictions(preds, top=5)


print('Predicted image labels:')
class_names, confidences = [], []
for class_id, class_name, confidence in decode_predictions(preds, top=5)[0]:
    print("{}(synset: {}): {:0.3f})".format(class_name, class_id, confidence))
# All keras pretrained vision models expect images with float 32 dtype and value in [0,255] range.

def classify_image(image):

    image.shape
    image.dtype
    image_224 = resize(image, (224, 224), preserve_range=True, mode = 'reflect')
    image_224.shape
    image_224.dtype
    image_224 = image_224.astype(np.float32)
    image_224.dtype
    image_224_batch = np.expand_dims(image_224, axis = 0)
    image_224_batch.shape
    x = preprocess_input(image_224_batch.copy())
    preds = model.predict(x)
    type(preds)
    preds.dtype, preds.shape, preds.sum(axis=1)
    return preds

image = imread('sofa.jpeg')
plt.imshow(image)
preds = classify_image(image)
decode_predictions(preds, top=5)
# Use other pretrained models to make the predictions
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.applications.inception_resnet_v2 import decode_predictions
model = InceptionResNetV2(weights='imagenet')

import tensorflow.keras.backend as K
K.image_data_format()
model.input_shape


def classify_image(image):

    image.shape
    image.dtype
    image_299 = resize(image, (299, 299), preserve_range=True, mode = 'reflect')
    image_299.shape
    image_299.dtype
    image_299 = image_299.astype(np.float32)
    image_299.dtype

    image_299_batch = np.expand_dims(image_299, axis = 0)
    image_299_batch.shape
    x = preprocess_input(image_299_batch.copy())
    preds = model.predict(x)

    type(preds)
    preds.dtype, preds.shape, preds.sum(axis=1)
    return preds

image = imread('lamp.jpeg')
plt.imshow(image)
preds = classify_image(image)
decode_predictions(preds, top=5)
# Inception Net V2 has higher confidence while classifying items but accuracy is not necessarity high

from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.applications.mobilenet import decode_predictions
model = MobileNet(weights='imagenet')

import tensorflow.keras.backend as K
K.image_data_format()
model.input_shape


def classify_image(image):

    image.shape
    image.dtype
    image_299 = resize(image, (224, 224), preserve_range=True, mode = 'reflect')
    image_299.shape
    image_299.dtype
    image_299 = image_299.astype(np.float32)
    image_299.dtype

    image_299_batch = np.expand_dims(image_299, axis = 0)
    image_299_batch.shape
    x = preprocess_input(image_299_batch.copy())
    preds = model.predict(x)

    type(preds)
    preds.dtype, preds.shape, preds.sum(axis=1)
    return preds

image = imread('Image.jpeg')
plt.imshow(image)
preds = classify_image(image)
decode_predictions(preds, top=5)
