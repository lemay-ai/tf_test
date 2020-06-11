# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import gravityai as grav

# Load training dataPath
data=pd.read_csv(grav.getInputFile())

# Define model 
model = keras.models.load_model('inference.h5')

x_test = np.array(data[['col1','col2','col3','col4']])
x_test=x_test/100
predictions = np.around(model.predict(x_test))
print(predictions.shape)
#print(predictions)
data['output1'] = predictions
data['output2'] = np.ones(predictions.shape)-predictions
data['output1']=data['output1'].astype(np.int)
data['output2']=data['output2'].astype(np.int)
data.to_csv(path_or_buf=grav.getOutputFile(), index=False)
