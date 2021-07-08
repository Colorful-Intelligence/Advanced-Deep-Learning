#%% Import Libraries
from keras.models import Sequential
from keras.layers import Conv2D , MaxPooling2D,Activation,Dropout,Flatten,Dense,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop,Adam
from keras.callbacks import ReduceLROnPlateau
import seaborn as sns

import matplotlib.pyplot as plt
from glob import glob

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

#%% Read the dataset

train = pd.read_csv("mnist_train.csv")
test = pd.read_csv("mnist_test.csv")

#%% Exploratory Data Analysis (EDA)

# count plot 
print(train.label.value_counts())
plt.figure(figsize = (10,10))
sns.countplot(train.label,palette = "icefire")


#%% Train - Test Split

y_train = train.label
x_train = train.drop(labels = ["label"],axis = 1) # axis = 1 , which means column


y_test = test.label
x_test = test.drop(labels = ["label"],axis = 1) # axis = 1 , which means column

#%% Plotting some of example

plt.subplot(3,2,1)
img1 = x_train.iloc[0].to_numpy().reshape((28,28))
plt.axis("off")
plt.imshow(img1,cmap = "gray")

plt.subplot(3,2,2)
img2 = x_train.iloc[15].to_numpy().reshape((28,28))
plt.axis("off")
plt.imshow(img2,cmap = "gray")


plt.subplot(3,2,3)
img3 = x_train.iloc[125].to_numpy().reshape((28,28))
plt.axis("off")
plt.imshow(img3,cmap = "gray")



#%% Normalization

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train = x_train/255.0
x_test = x_test/255.0

#%% Reshaping
x_train = x_train.values.reshape(-1,28,28,1)
x_test = x_test.values.reshape(-1,28,28,1)

#%% Label Encoding 

y_train = to_categorical(y_train,num_classes=10)
y_test = to_categorical(y_test,num_classes=10)

#%% Train - Test Split

test_size = 0.15

from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size = test_size,random_state=42)



epochs = 75
batch_size = 240
num_of_classes = 10

model = Sequential()

model.add(Conv2D(filters = 128, kernel_size = (3,3), padding ="same",
                 activation ="relu", input_shape =(28,28,1)))
model.add(MaxPooling2D(pool_size =(3,3)))

model.add(Conv2D(64,3,3))
model.add(Activation("relu"))
model.add(MaxPooling2D(3,3))

model.add(Flatten())
model.add(Dense(1024))  #Hidden layer1
model.add(Activation("relu"))
model.add(Dropout(0.25))

model.add(Dense(num_of_classes)) # Output layer size must equal to number of classes (labels)
model.add(Activation("softmax"))


#%% Learning Rate Optimizer
learning_rate_optimizer = ReduceLROnPlateau(monitor = "val_accuracy",
                                           patience = 2, verbose = 1,
                                           factor = 0.5, min_lr = 0.000001)


#%% Compiling Model
optimizer = RMSprop()
model.compile(optimizer = optimizer, loss  ="categorical_crossentropy", metrics =["accuracy"])




#%% Data Augmentation

datagen = ImageDataGenerator( 
        shear_range = 0.2,
        zoom_range = 0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip = True,
        vertical_flip = True)

datagen.fit(x_train)


history = model.fit(datagen.flow(x_train,y_train, 
                                batch_size = batch_size), 
                                epochs = epochs,
                                validation_data = (x_val,y_val),
                                steps_per_epoch = x_train.shape[0]//batch_size,
                                callbacks = [learning_rate_optimizer])


#%% Test Data Results

score = model.evaluate(x_test, y_test, verbose = 0)
print("Test Loss : %f \nTest Accuracy : %f "%(score[0],score[1]))

#%% Model Evaluation

print(history.history.keys())
plt.plot(history.history["loss"], label ="Train Loss")
plt.plot(history.history["val_loss"], label ="Test Loss")
plt.legend()
plt.show()

#-----------------------------------------------------------------------

print(history.history.keys())
plt.plot(history.history["accuracy"], label ="Train Accuracy")
plt.plot(history.history["val_accuracy"], label ="Test Accuracy")
plt.legend()
plt.show()

