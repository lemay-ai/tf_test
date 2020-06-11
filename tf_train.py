# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import gravityai as grav

# Load training dataPath
data=pd.read_csv(grav.getInputFile())
Y = np.array(data[['output1']])
X = np.array(data[['col1','col2','col3','col4']])
X=X/100
print(X.shape,Y.shape)
# Define model
model = tf.keras.Sequential()
model.add(layers.Dense(4, activation='relu'))

model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

#compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Set a callback (from: https://medium.com/iitg-ai/how-to-use-callbacks-in-keras-to-visualize-monitor-and-improve-your-deep-learning-model-c9ca37901b28)
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc') >= 1.0):
          print("\nWe have reached %2.2f%% accuracy, so we will stopping training." %(1.0*100))
          self.model.stop_training = True

callbacks = myCallback()

# fit model to data
model.fit(X,Y,epochs=100000000,batch_size=100,callbacks=[callbacks])

x_test = np.array(data[['col1','col2','col3','col4']])
x_test=x_test/100
predictions = np.around(model.predict(x_test))
print(predictions.shape)
print(predictions)
data['output1'] = predictions
data['output2'] = np.ones(predictions.shape)-predictions
data['output1']=data['output1'].astype(np.int)
data['output2']=data['output2'].astype(np.int)
data.to_csv(path_or_buf=grav.getOutputFile(), index=False)
model.save("inference.h5")

