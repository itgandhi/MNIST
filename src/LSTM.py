
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
	
	Y = df['label'].values	

	df=df.loc[:,df.columns != 'label']	

	X = (df.values).reshape(df.values.shape[0],28,28)
	return(X,Y)

def scale(X, eps = 0.001):
	return (X-np.min(X,axis=0))/(np.max(X,axis=0)+eps)

def nudge(X,Y):
	translations = [(0,-1),(0,1),(-1,0),(1,0)]
	data = []
	target = []

	for(image, label) in zip(X,Y):
		 
		for(tX,tY) in translations:
			M = np.float32([[1,0,tX],[0,1,tY]])
			data.append(cv2.warpAffine(image,M,(28,28)).reshape(-1,1))
			target.append(label)
	
	return(np.array(data),np.array(target))

def lstm(X_trainShape):

	hidden_units = int(50) 


	clip_norm = float(5.0)
	forget_bias = 1.0

	model = Sequential()
	model.add(LSTM(hidden_units, input_shape=X_trainShape, inner_init='glorot_uniform',
	forget_bias_init='one', activation='tanh', inner_activation='sigmoid'))
	model.add(Dense(10))
	model.add(Activation('softmax'))
	rmsprop = RMSprop( clipnorm=clip_norm)
	model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])

	return model


ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required = True, help = "path to dataset file")
ap.add_argument("-t","--test",required = True, type = float, help = "size of test split. eg. 0.4 = 60% train 40% test...")

ap.add_argument("-m","--modelpath",required = True, help = "path to save trained model")

args = vars(ap.parse_args())


(X,Y) = load_data(args['dataset'])
X = X.astype("float32")
X = scale(X)


(X,Y) = nudge(X,Y)

(trainX, testX, trainY, testY) = train_test_split(X,Y,test_size = args['test'], random_state = 42)	



print("shape X : ",trainX.shape)
print("shape Y : ",trainY.shape)


trainY = np_utils.to_categorical(trainY)
testY = np_utils.to_categorical(testY)
num_classes = testY.shape[1]


model = lstm(trainX.shape[1:])

checkpoint = ModelCheckpoint(args['modelpath']+'/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

model.fit(trainX, trainY, validation_data=(testX,testY), epochs=1000, batch_size=100,verbose=1, callbacks=[checkpoint])

scores = model.evaluate(testX,testY, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))


