
import numpy as np
import pandas as pd
import cv2
import argparse


from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.optimizers import RMSprop
from keras.utils import np_utils
import sys, os
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras import backend as K
K.set_image_dim_ordering('th')



def load_data(dataPath):
	df = pd.read_csv(dataPath)
	X = (df.values).reshape(df.values.shape[0],-1,1)
	return(X)

def scale(X, eps = 0.001):
	return (X-np.min(X,axis=0))/(np.max(X,axis=0)+eps)

def lstm(modelpath,X_trainShape):

	hidden_units = int(50) 


	clip_norm = float(5.0)
	forget_bias = 1.0

	model = Sequential()
	model.add(LSTM(hidden_units, input_shape=X_trainShape, inner_init='glorot_uniform',
	forget_bias_init='one', activation='tanh', inner_activation='sigmoid'))
	model.add(Dense(10))
	model.add(Activation('softmax'))
	rmsprop = RMSprop( clipnorm=clip_norm)
	model.load_weights(modelpath)
	model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])

	return model


ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required = True, help = "path to dataset file")

ap.add_argument("-m","--modelpath",required = True, help = "path to save trained model")

ap.add_argument("-s","--submissionpath",required = True , help = "path where you want to store sbmission file..", default = 0)
args = vars(ap.parse_args())


X = load_data(args['dataset'])
X = X.astype("float32")
X = scale(X)

print("shape X : ",X.shape)
#exit(0)

model = lstm(args['modelpath'],X.shape[1:])

result = model.predict(X)
result = result.argmax(1)

d = {'ImageId':range(1,X.shape[0]+1),'Label':result}
df = pd.DataFrame(data=d)
df.set_index('ImageId',inplace=True)
df.to_csv(args['submissionpath'])

