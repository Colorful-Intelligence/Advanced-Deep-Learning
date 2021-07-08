#%% Libraries
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation,Dropout,Flatten,Dense 
from keras.preprocessing.image import  ImageDataGenerator,img_to_array,load_img
import matplotlib.pyplot as plt
from glob import glob

#%% Read dataset

train_path = "fruits-360/Training/" # train dataset okuduk
test_path = "fruits-360/Test/" # test dataset okuduk

img = load_img(train_path + "Apple Braeburn/0_100.jpg") # burada örnek olması için dataset içindeki bir resmi gösterdik
plt.imshow(img)
plt.axis("off")
plt.show()

x = img_to_array(img) # resmi matrix'e çevirdik
print(x.shape)

className = glob(train_path + "/*") # train dosyası içindeki class isimlerini aldık

numberOfClass = len(className)
print("Number of Class: ",numberOfClass)



#%% CNN Model

model = Sequential()

model.add(Conv2D(filters=32 , kernel_size=(3,3), input_shape = x.shape))
model.add(Activation("relu")) # Aktivasyon fonksiyonu
model.add(MaxPooling2D())

model.add(Conv2D(filters=32 , kernel_size=(3,3)))
model.add(Activation("relu")) # Aktivasyon fonksiyonu
model.add(MaxPooling2D())

model.add(Conv2D(filters=64 , kernel_size=(3,3)))
model.add(Activation("relu")) # Aktivasyon fonksiyonu
model.add(MaxPooling2D())

# Flatten Operation
model.add(Flatten())
model.add(Dense(1024)) # 1024 is a number of neural
model.add(Activation("relu"))
model.add(Dropout(0.5)) # her seferiinde nöronların %50 kapalı oluyor
model.add(Dense(numberOfClass)) #output (output sayımız class sayımız kadar olmalı !!)
model.add(Activation("softmax"))


model.compile(loss = "categorical_crossentropy",
              optimizer = "rmsprop",
              metrics = ["accuracy"])

batch_size = 32


#%% Data Generation - Train - Test

train_datagen = ImageDataGenerator(rescale = 1./255,
                   shear_range = 0.3,
                   horizontal_flip=True,
                   zoom_range=0.3)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(train_path,
                                                    target_size = x.shape[:2],
                                                    batch_size= batch_size,
                                                    color_mode = "rgb",
                                                    class_mode = "categorical"
                                                    )


test_generator = test_datagen.flow_from_directory(test_path,
                                                    target_size = x.shape[:2],
                                                    batch_size= batch_size,
                                                    color_mode = "rgb",
                                                    class_mode = "categorical"
                                                    )



# Modeli Fit Etme (Eğitme)
hist = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=1600 // batch_size,
    epochs = 100,
    validation_data=test_generator,
    validation_steps=800 // batch_size)

#%% Model Save
model.save_weights("Fruits_weights.h5")

#%% Model Evaluation
print(hist.history.keys())
plt.plot(hist.history["loss"],label = "Train Loss")
plt.plot(hist.history["val_loss"],label = "Validation Loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(hist.history["accuracy"],label = "Train accuracy")
plt.plot(hist.history["val_accuracy"],label = "Validation accuracy")
plt.legend()
plt.show()






































































