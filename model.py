from cv2 import cv2
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os


from tensorflow.keras.preprocessing.image import ImageDataGenerator

face_clsfr = cv2.CascadeClassifier("C:\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")


train_dir = 'Train'
test_dir = 'Test'
val_dir = 'Validation'



train_datagen = ImageDataGenerator(rescale=1.0/255, horizontal_flip=True, zoom_range=0.2,shear_range=0.2)
train_generator = train_datagen.flow_from_directory(directory=train_dir,target_size=(128,128),class_mode='categorical',batch_size=32)

val_datagen = ImageDataGenerator(rescale=1.0/255)
val_generator = train_datagen.flow_from_directory(directory=val_dir,target_size=(128,128),class_mode='categorical',batch_size=32)

test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = train_datagen.flow_from_directory(directory=val_dir,target_size=(128,128),class_mode='categorical',batch_size=32)


model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (128, 128, 3)),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Dropout(0.25),
                                    tf.keras.layers.Conv2D(64, (3, 3), activation = "relu"),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Dropout(0.25),
                                    tf.keras.layers.Conv2D(128, (3, 3), activation = "relu"),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Dropout(0.25),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(2, activation = "sigmoid")])


model.summary()

model.compile(optimizer = RMSprop(lr=0.001), loss = "categorical_crossentropy", metrics=["accuracy"])

class myCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs = {}):
          if(logs.get('accuracy') >= 0.98):
              self.model.stop_training = True

callbacks = myCallback()
model.fit(train_generator, epochs=5, validation_data = val_generator, callbacks = [callbacks])

labels_dict={0:'NO MASK !', 1:'MASK'}
color_dict={0:(0,0,255),1:(0,255,0)}

rect_size = 4
cap = cv2.VideoCapture(0) 


while True:
    (ret, img) = cap.read()
    img = cv2.flip(img, 1, 1)

    resized = cv2.resize(img,(128,128))
    faces = face_clsfr.detectMultiScale(resized)
    for f in faces:
        (x, y, w, h) = [v * rect_size for v in f] 
        
        face_img = img[y:y+h, x:x+w]
        resized=cv2.resize(face_img,(128,128))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,128,128,3))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)

        
        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    cv2.imshow('MASKDETECTION',   img)
    key = cv2.waitKey(10)
    
    if key == 27:   # Esc
        break

cap.release()

cv2.destroyAllWindows()
