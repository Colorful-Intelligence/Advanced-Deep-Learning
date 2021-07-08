#%% Import Libraries
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.utils import to_categorical
from keras.applications.vgg19 import VGG19
import matplotlib.pyplot as plt
from glob import glob
import numpy as np

import cv2

import warnings
warnings.filterwarnings("ignore")


#%% Read the dataset

from keras.datasets import cifar10
(x_train,y_train),(x_test,y_test)  = cifar10.load_data()
print(x_train.shape)
print(x_train[0])

numberOfClass = 10
y_train = to_categorical(y_train,numberOfClass)
y_test = to_categorical(y_test,numberOfClass)

input_shape = x_train.shape[1:]

# Visuzalize example of the data

plt.imshow(x_train[5548].astype(np.uint8))
plt.axes("off")
plt.show()

plt.imshow(x_train[2569].astype(np.uint8))
plt.axes("off")
plt.show()

#%% Increase Diemension

def resize_img(img):
    numberOfImage = img.shape[0]
    new_array = np.zeros((numberOfImage, 48,48,3)) # burada 32x32'lik resimleri 48x48'lik yaptık ve RBG olması için 3 dedik.
    for i in range(numberOfImage):
        new_array[i] = cv2.resize(img[i,:,:,:],(48,48)) # [i,:,:,:] 32*32,3 burada resimdeki tüm boyutları aldık.
    return new_array


x_train = resize_img(x_train)
x_test = resize_img(x_test)
print("inceased dimension x_train : ",x_train.shape)

plt.figure()
plt.imshow(x_train[5368].astype(np.uint8))
plt.axes("off")
plt.show()

#%% VCC19

vgg = VGG19(include_top=False,weights="imagenet",input_shape=(48,48,3)) #inckude_top = False , which means we did not add fully connected layer.
print(vgg.summary())


vgg_layer_list = vgg.layers
print(vgg_layer_list)

model = Sequential()
for layer in vgg_layer_list:
    model.add(layer)

print(model.summary())

for layer in model.layers:
    layer.trainable = False

# Fully con Layers
model.add(Flatten())
model.add(Dense(128)) # 128 = number of neurons
model.add(Dense(numberOfClass,activation = "softmax"))


print(model.summary())


model.compile(loss = "categorical_crossentropy",
              optimizer = "rmsprop",
              metrics = ["accuracy"]
              )






hist = model.fit(x_train,y_train,validation_split=0.2,epochs = 2,batch_size = 500)



#%%  model save
model.save_weights("example.h5")

#%%
plt.plot(hist.history["loss"], label = "train loss")
plt.plot(hist.history["val_loss"], label = "val loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(hist.history["accuracy"], label = "train accuracy")
plt.plot(hist.history["val_accuracy"], label = "val accuracy")
plt.legend()
plt.show()




