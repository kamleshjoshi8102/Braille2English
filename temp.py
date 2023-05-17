import pandas as pd
import numpy as np
import cv2
import urllib.request
import matplotlib.pyplot as plt
import string
import random

from tensorflow import keras
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout,Flatten
from keras.layers import MaxPooling2D,Conv2D

from keras import layers as L
import numpy as np

model= load_model('braille_train.h5')

img_path = "world.png"
img = cv2.imread(img_path)
cv2.imshow("zz",img)

h,w,c = img.shape
print("height: ",h , " width : ", w) # h,w

img_no=cv2.resize(img,(w,116))
cv2.imshow("zz",img_no)
h1,w1,c1=img_no.shape

sentence=""
alphabet = list(string.ascii_lowercase)
cur_pos = 0
target = {}
for letter in alphabet:
    target[letter] = [0] * 27
    target[letter][cur_pos] = 1
    cur_pos += 1
target[' '] = [0] * 27
target[' '][26] = 1


for wid in range (0,w1,72):
  img_crop=img_no[:, 0+wid:72+wid]
  cv2.imshow("zz",img_crop)
  pred_img = cv2.resize(img_crop, (28,28),interpolation=cv2.INTER_CUBIC)   
  cv2.imshow("zz",pred_img)
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