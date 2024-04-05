import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import cv2
import glob
from PIL import Image
import re


# ts = pd.read_csv('train.csv')
# #
# down_w = 128
# down_h = 128
# down_sz = (down_w,down_h)
# #
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# for i in range(0,69540):
#   print(str(i))
#   img = np.asarray(Image.open('train_1/' + str(i) + '.jpg'))
#   print(img.shape)
#   if (img.shape[-1] != 3 and len(img.shape) == 3):
#     ts = ts.drop(i)
#     print(str(img.shape[-1]) + " layers")
#     print("Row " + str(i) + " dropped!")
#     continue
#   img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#   faces = face_cascade.detectMultiScale(gray,1.3,9)
#   im1 = 0
#   for (x, y, w, h) in faces:
#     c_img = img[y:y+h,x:x+w]
#     im1 = c_img
#   if (type(im1) == int):
#     resz = cv2.resize(img,down_sz,interpolation=cv2.INTER_LINEAR)
#     print("whole")
#   else:
#     resz = cv2.resize(im1,down_sz,interpolation=cv2.INTER_LINEAR)
#     print("crop")
#   cv2.imwrite('cropped/' + str(i) + ".jpg", resz)
#   print('cropped/' + str(i) + ".jpg")
# ts.to_csv('final_train.csv')
# #
# for f in glob.iglob('test/*'):
#   print(f)
#   img = np.asarray(Image.open(f))
#   img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#   faces = face_cascade.detectMultiScale(gray,1.3,9)
#   im1 = 0
#   for (x, y, w, h) in faces:
#     c_img = img[y:y+h,x:x+w]
#     im1 = c_img
#   numbers = re.findall('\d+', f)
#   numbers = int(numbers[-1])
#   if (type(im1) == int):
#     resz = cv2.resize(img,down_sz,interpolation=cv2.INTER_LINEAR)
#     print("whole")
#   else:
#     resz = cv2.resize(im1,down_sz,interpolation=cv2.INTER_LINEAR)
#     print("crop")
#   cv2.imwrite('cropped_test/' + str(numbers) + ".jpg", resz)
#   print('cropped_test/' + str(numbers) + ".jpg")



x_test = []
for i in range(0,4977):
    img = np.asarray(Image.open('cropped_test/' + str(i) + '.jpg'))
    x_test.append(img)
x_test = np.array(x_test)

ts = pd.read_csv('final_train.csv')
vals = ts['Unnamed: 0']
vals = vals.tolist()
names = ts['Category']
names = names.tolist()
print(names[0])
print(vals[60:63])
print(names[60:63])

x_train = []
y_train = []
for i in vals:
    ind = vals.index(i)
    img = np.asarray(Image.open('train_1/' + str(i) + '.jpg'))
    x_train.append(img)
    y_train.append(names[ind])
x_train = np.array(x_train)
y_train = np.array(y_train)
y_train = y_train.reshape(len(vals),1)
print(y_train.shape)

encoder = OneHotEncoder(sparse_output=False)
y_tr = encoder.fit_transform(y_train)
print(np.argmax(y_tr[0]))
print(y_tr.shape)


num_classes = 100
input_shape = (128, 128, 3)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

batch_size = 128
epochs = 20

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_tr, batch_size=batch_size, epochs=epochs, validation_split=0.2)

y_test = model.predict(x_test)
y_out = encoder.inverse_transform(y_test)
y_out = y_out.tolist()

print(y_out)

output_names = []
for s in range(0,4977):
    name = y_out[s]
    output_names.append(name[0])
print(output_names)
df = pd.DataFrame(output_names)
df.columns = ['Category']
print(df)
df.to_csv("submission.csv")

