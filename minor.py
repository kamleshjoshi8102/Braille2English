import pandas as pd
import numpy as np
import cv2
import urllib.request
import matplotlib.pyplot as plt
import string
import random


import keras
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout,Flatten
from keras.layers import MaxPooling2D,Conv2D
#from keras.layers.advanced_activations import L.LeakyReLU
from keras import layers as L

df = pd.read_csv("./braille_data.csv")


alphabet = list(string.ascii_lowercase)
cur_pos = 0
target = {}
for letter in alphabet:
    target[letter] = [0] * 27
    target[letter][cur_pos] = 1
    cur_pos += 1
target[' '] = [0] * 27
target[' '][26] = 1
    
data = []
for i, row in df.iterrows():
    picture = []
    url = row['Labeled Data']
    label = row['Label']
    cur_target = target[label[11]]
    x = urllib.request.urlopen(url)
    resp = x.read()
    image = np.array(bytearray(resp), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (28,28))
    image = image.astype(np.float32)/255.0
    picture.append(image)
    picture.append(cur_target)
    data.append(picture)

#  It is taking the data and splitting it into two parts:
#     a. The first part is the input data, which is the image
#     b. The second part is the output data, which is the label
length = len(data)
for i in range(length):
    if(data[i][1][0] == 1):
        data[i][1] = [0] * 27
        data[i][1][0] = 1
    elif(data[i][1][1] ==1):
        data[i][1] =[0] * 27
        data[i][1][1] =1
    elif(data[i][1][2] ==1):
        data[i][1] =[0] * 27
        data[i][1][2] =1
    elif(data[i][1][3] ==1):
        data[i][1] =[0] * 27
        data[i][1][3] =1
    elif(data[i][1][4] ==1):
        data[i][1] =[0] * 27
        data[i][1][4] =1
    elif(data[i][1][5] ==1):
        data[i][1] =[0] * 27
        data[i][1][5] =1
    elif(data[i][1][6] ==1):
        data[i][1] =[0] * 27
        data[i][1][6] =1
    elif(data[i][1][7] ==1):
        data[i][1] =[0] * 27
        data[i][1][7] =1
    elif(data[i][1][8] ==1):
        data[i][1] =[0] * 27
        data[i][1][8] =1
    elif(data[i][1][9] ==1):
        data[i][1] =[0] * 27
        data[i][1][9] =1
    elif(data[i][1][10] ==1):
        data[i][1] =[0] * 27
        data[i][1][10] =1
    elif(data[i][1][11] ==1):
        data[i][1] =[0] * 27
        data[i][1][11] =1
    elif(data[i][1][12] ==1):
        data[i][1] =[0] * 27
        data[i][1][12] =1
    elif(data[i][1][13] ==1):
        data[i][1] =[0] * 27
        data[i][1][13] =1
    elif(data[i][1][14] ==1):
        data[i][1] =[0] * 27
        data[i][1][14] =1
    elif(data[i][1][15] ==1):
        data[i][1] =[0] * 27
        data[i][1][15] =1
    elif(data[i][1][16] ==1):
        data[i][1] =[0] * 27
        data[i][1][16] =1
    elif(data[i][1][17] ==1):
        data[i][1] =[0] * 27
        data[i][1][17] =1
    elif(data[i][1][18] ==1):
        data[i][1] =[0] * 27
        data[i][1][18] =1
    elif(data[i][1][19] ==1):
        data[i][1] =[0] * 27
        data[i][1][19] =1
    elif(data[i][1][20] ==1):
        data[i][1] =[0] * 27
        data[i][1][20] =1
    elif(data[i][1][21] ==1):
        data[i][1] =[0] * 27
        data[i][1][21] =1
    elif(data[i][1][22] ==1):
        data[i][1] =[0] * 27
        data[i][1][22] =1
    elif(data[i][1][23] ==1):
        data[i][1] =[0] * 27
        data[i][1][23] =1
    elif(data[i][1][24] ==1):
        data[i][1] =[0] * 27
        data[i][1][24] =1
    elif(data[i][1][25] ==1):
        data[i][1] =[0] * 27
        data[i][1][25] =1
    elif(data[i][1][26] ==1):
        data[i][1] =[0] * 27
        data[i][1][26] =1

#creates array to test, train and validate the model 

random.shuffle(data)

data = np.asarray(data)
train_dataset = data[:1124]
test_dataset = data[1124:1264]
valid_dataset = data[1264:1404]

train_dataset_img = np.array(train_dataset[:,0])
train_dataset_label = np.array(train_dataset[:,1])
test_dataset_img = np.array(test_dataset[:,0])
test_dataset_label = np.array(test_dataset[:,1])
valid_dataset_img = np.array(valid_dataset[:,0])
valid_dataset_label = np.array(valid_dataset[:,1])

#to expand dimension

a=np.expand_dims(train_dataset_img[0],axis=0)
b=np.expand_dims(train_dataset_img[1],axis=0)
tr_ds_img=np.append(a,b,axis=0)
for i in range(2,1124):
    x=np.expand_dims(train_dataset_img[i],axis=0)
    tr_ds_img=np.append(tr_ds_img,x,axis=0)
    
a1=np.expand_dims(test_dataset_img[0],axis=0)
b1=np.expand_dims(test_dataset_img[1],axis=0)
ts_ds_img=np.append(a1,b1,axis=0)
for i in range(2,140):
    x1=np.expand_dims(test_dataset_img[i],axis=0)
    ts_ds_img=np.append(ts_ds_img,x1,axis=0)

a2=np.expand_dims(valid_dataset_img[0],axis=0)
b2=np.expand_dims(valid_dataset_img[1],axis=0)
va_ds_img=np.append(a2,b2,axis=0)
for i in range(2,140):
    x=np.expand_dims(valid_dataset_img[i],axis=0)
    va_ds_img=np.append(va_ds_img,x,axis=0)

tr_ds_lb = np.expand_dims(train_dataset_label[0],axis=0)
for i in range(1,1124):
    x3 = np.expand_dims(train_dataset_label[i],axis=0)
    tr_ds_lb = np.append(tr_ds_lb,x3,axis=0)

ts_ds_lb = np.expand_dims(test_dataset_label[0],axis=0)
for i in range(1,140):
    x4 = np.expand_dims(test_dataset_label[i],axis=0)
    ts_ds_lb = np.append(ts_ds_lb,x4,axis=0)
    
va_ds_lb = np.expand_dims(valid_dataset_label[0],axis=0)
for i in range(1,140):
    x5 = np.expand_dims(valid_dataset_label[i],axis=0)
    va_ds_lb = np.append(va_ds_lb,x5,axis=0)

epochs= 20 #no of iteration over data set 
batch_size = 32 #number of training examples that are processed at a time by the model after these examples are evaluated the parameters are updated
num_classes = 27 #output shape of the model, ie the o/p layer will have 27 neurons  

braille_model = Sequential()
braille_model.add(Conv2D(16, kernel_size=(5,5), activation='linear', input_shape=(28,28,3), padding='same', strides=1))
braille_model.add(L.LeakyReLU(alpha = 0.1))
braille_model.add(MaxPooling2D((2,2)))
braille_model.add(Conv2D(32, kernel_size=(5,5), activation ='linear', padding='same', strides=1))
braille_model.add(L.LeakyReLU(alpha = 0.1))
braille_model.add(MaxPooling2D(pool_size=(2,2)))
braille_model.add(Dropout(0.25)) #dropout accuracy reduced to 93
braille_model.add(Conv2D(64, kernel_size=(5,5), activation='linear', padding='same',strides=1))
braille_model.add(L.LeakyReLU(alpha=0.1))
braille_model.add(MaxPooling2D(pool_size=(2,2)))
braille_model.add(Conv2D(128, kernel_size=(5,5), activation='linear', padding='same',strides=1))
braille_model.add(L.LeakyReLU(alpha=0.1))
braille_model.add(MaxPooling2D(pool_size=(2,2)))
braille_model.add(Flatten())
braille_model.add(Dense(256,activation='linear'))
braille_model.add(L.LeakyReLU(alpha=0.1))
braille_model.add(Dense(num_classes,activation='softmax'))

braille_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

#fit the model
braille_train = braille_model.fit(tr_ds_img ,tr_ds_lb , batch_size=batch_size , epochs=epochs, verbose=1, validation_data=(va_ds_img, va_ds_lb))

test_eval = braille_model.evaluate(ts_ds_img,ts_ds_lb,verbose=1)

braille_model.save('braille_train.h5')


braille_model.summary()

import numpy as np

model= load_model('braille_train.h5')

train_loss, train_acc = model.evaluate(tr_ds_img, tr_ds_lb)
test_loss, test_acc = model.evaluate(ts_ds_img, ts_ds_lb)
val_loss, val_acc = model.evaluate(va_ds_img, va_ds_lb)

print("Train Accuracy: ", train_acc*100)
print("validation  Accuracy: ", val_acc*100)
print("Test Accuracy: ", test_acc*100)

history=braille_train.history
plt.plot(history['loss'], 'red', label='Training loss')
plt.plot(history['val_loss'], 'blue', label='Validation loss')
plt.legend(loc='best')
plt.title("Learning curve")
plt.show()

plt.plot(history['accuracy'], 'r', label='Training accuracy')
plt.plot(history['val_accuracy'], 'b', label='Validation accuracy')
plt.legend(loc='best')
plt.title("Accuracy curve")
plt.show()


img_path="world.png"
img = cv2.imread(img_path)
cv2_imshow(img)

h,w,c = img.shape
print("height: ",h , " width : ", w) # h,w


img_no=cv2.resize(img,(w,116))
cv2_imshow(img_no)
h1,w1,c1=img_no.shape


sentence=""
model= load_model('braille_train.h5')

for wid in range (0,w1,72):
  img_crop=img_no[:, 0+wid:72+wid]
  cv2_imshow(img_crop)
  pred_img = cv2.resize(img_crop, (28,28),interpolation=cv2.INTER_CUBIC)   
  cv2_imshow(pred_img)
  pred_img = pred_img.astype(np.float32)/255.0
  pred_img = np.expand_dims(pred_img,axis=0)
  pred_lb = model.predict(pred_img)
  for j in range(len(pred_lb[0])):
          if pred_lb[0][j] > 0.6:
              pred_lb[0][j] = 1.0
          else:
              pred_lb[0][j] = 0.0
  for key,value in target.items():
          if np.array_equal(np.asarray(pred_lb[0]),np.asarray(value)):
            print(key)
            sentence=sentence+key

print(sentence)