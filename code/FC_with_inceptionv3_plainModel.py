import numpy as np
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
import tensorflow as tf

from tensorflow.keras import layers,models
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import classification_report, confusion_matrix

import seaborn as sns


#select pretrained model as InceptionV3
pretrainedModel = InceptionV3(input_shape=(299,299,3), weights="imagenet", include_top=False)
for layer in pretrainedModel.layers:
  layer.trainable = False  
# pretrainedModel.summary()
last_output = pretrainedModel.output
x = layers.GlobalAveragePooling2D()(last_output)   #added a new layer of Global average pooling
x = layers.Dense(1024, activation='relu')(x)       #added a dense layer with "relu" activation function
x = layers.Dropout(0.25)(x)                         
x = layers.Dense(6, activation='softmax')(x)       #added a last layer softmax layer. the number of nodes are 6 which are equal to output classes

model = Model(pretrainedModel.input, x)

model.compile(loss = 'categorical_crossentropy',
              optimizer ='adam',
              metrics=['accuracy'])

model.summary()

# create a array of classes names
filenameAll = os.listdir("/data/AllPhotos") # dataset path.
categoryAll = []
for filename in filenameAll:
  if("Apple" in filename):
    categoryAll.append("Apple")
  if("Banana" in filename):
    categoryAll.append("Banana")
  if("Guava" in filename):
    categoryAll.append("Guava")
  if("Lime" in filename):
    categoryAll.append("Lime")
  if("Orange" in filename):
    categoryAll.append("Orange")
  if("Pomegranate" in filename):
    categoryAll.append("Pomegranate")

classifyOnlyFruitdf = pd.DataFrame({
    'filename': filenameAll,
    'category': categoryAll
})

classifyOnlyFruitdf = classifyOnlyFruitdf.sort_values(by ='category', ascending = 1)

classifyOnlyFruitdf['category'].value_counts().plot.bar()

train_df, test_df = train_test_split(classifyOnlyFruitdf, test_size=0.20, random_state=42) #cosider 80:20 for training and testing
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

total_train = train_df.shape[0]
total_validate = test_df.shape[0]
batch_size=30

train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

#map the data frames and Images
train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "/data/AllPhotos", 
    x_col="filename",
    y_col="category",
    target_size=(299, 299),
    class_mode="categorical",
    batch_size=batch_size
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    test_df, 
    "/data/AllPhotos", 
    x_col='filename',
    y_col='category',
    target_size=(299, 299),
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False
)

#Train the model 
history = model.fit(
    train_generator, 
    epochs=5,
    validation_data=test_generator,
    validation_steps=total_validate/batch_size,
    steps_per_epoch=total_train/batch_size,
)

# Code to test the model 

classes = ['Apple','Banana', 'Guava', 'Lime', 'Orange', 'Pomegranate']

img_path = '/data/TestingImages/AppleGood21.jpg'
img = image.load_img(img_path, target_size=(299, 299, 3))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict([x])
print(classes[np.argmax(preds)])
print(preds)

model.save("results/FCInceptionV3_plainModel.h5")

#===============================================================
#Follwoing code can be used to print confusion matrix
#from sklearn.metrics import classification_report, confusion_matrix

#Y_pred = model.predict_generator(test_generator, total_validate // batch_size+1)
#y_pred = np.argmax(Y_pred, axis=1)
#labels = [1,2,3,4,5,6]
#print('Confusion Matrix')
#cm = confusion_matrix(test_generator.classes,y_pred)
#target_names=["Apple", "Banana", "Guava", "Lime", "Orange","Pomegranate"]

#import seaborn as sns
#import matplotlib.pyplot as plt

#ax = plt.subplot()
#fig_dims = (10,7)
#fig,ax = plt.subplots(figsize=fig_dims);

#sns.heatmap(cm/np.sum(cm),fmt=".2%",annot=True,ax = ax,cmap="Blues");

#labels, title and ticks
#sns.set(font_scale=1) # for label size
#ax.set_xlabel('Predicated Labels');ax.set_ylabel('True labels');
#ax.set_title('Confusion Matrix');
#ax.xaxis.set_ticklabels(target_names); ax.yaxis.set_ticklabels(target_names);