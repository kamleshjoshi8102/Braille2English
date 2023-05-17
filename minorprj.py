import pandas as pd
import numpy as np
import cv2
import urllib.request
import matplotlib.pyplot as plt
import string
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

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
    image = cv2.resize(image, (28, 28))
    image = image.astype(np.float32) / 255.0
    picture.append(image)
    picture.append(cur_target)
    data.append(picture)

random.shuffle(data)

data = np.asarray(data)
dataset_img = np.array(data[:, 0])
dataset_label = np.array(data[:, 1])

dataset_img = dataset_img.reshape(dataset_img.shape[0], -1)

X_train, X_test, y_train, y_test = train_test_split(dataset_img, dataset_label, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

train_predictions = model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)

test_predictions = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)

print("Train Accuracy:", train_accuracy * 100)
print("Test Accuracy:", test_accuracy * 100)
