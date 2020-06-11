# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
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
Y = data[['output1', 'output2']]
X = data[['col1','col2','col3','col4']]

# Define model 
model = tf.keras.Sequential()
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(2))

#compile model
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# fit model to data e
model.fit(X,Y)

x_test = data[['col1','col2','col3','col4']]
predictions = model.predict(x_test)
test_df['output1'] = predictions[0]
test_df['output2'] = predictions[1]
test_df.to_csv(path_or_buf=outPath, index=False)
print(test_df.to_csv(index=False))
