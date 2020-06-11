# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import warnings,sys,os
from pathlib import Path

warnings.filterwarnings("ignore")

if (len(sys.argv) <= 1):
    sys.stderr.write("No file specified on command line")
    sys.exit(2)

datafile = Path(sys.argv[1])
if (not datafile.is_file()):
    sys.stderr.write("Input file not found")
    sys.exit(2)

dataPath = str(datafile.resolve()) 

if (len(sys.argv) <= 2):
    sys.stderr.write("No output file specified on command line")
    sys.exit(3)

outfile = Path(sys.argv[2])
print(outfile)

outPath = str(outfile.absolute())
print(outPath)

print(__file__)
me = Path(__file__)
dir = me.parent

os.chdir(str(dir.resolve()))

# Load training dataPath
data=pd.read_csv(datafile)
Y = np.array(data[['output1']])
X = np.array(data[['col1','col2','col3','col4']])
X=X/100
print(X.shape,Y.shape)
# Define model 
model = tf.keras.Sequential()
model.add(layers.Dense(4, activation='relu'))

model.add(layers.Dense(1024, activation='relu'))
#model.add(layers.Dropout(0.5))
#model.add(layers.Dense(30, activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

#compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# fit model to data
model.fit(X,Y, epochs=5000,batch_size=100)

x_test = np.array(data[['col1','col2','col3','col4']])
x_test=x_test/100
predictions = np.around(model.predict(x_test))
print(predictions.shape)
print(predictions)
data['output1'] = predictions
data['output2'] = np.ones(predictions.shape)-predictions
data['output1']=data['output1'].astype(np.int)
data['output2']=data['output2'].astype(np.int)
data.to_csv(path_or_buf=outPath, index=False)
#print(data)
