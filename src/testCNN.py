import numpy as np
import pandas as pd
import cv2
import argparse


from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')



def load_data(dataPath):
	df = pd.read_csv(dataPath)
	X = (df.values).reshape(df.values.shape[0],1,28,28)
	return(X)

def scale(X, eps = 0.001):
	return (X-np.min(X,axis=0))/(np.max(X,axis=0)+eps)

def larger_model(modelpath):
	# create model
	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(10, activation='softmax'))
	model.load_weights(modelpath)
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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

model = larger_model(args['modelpath'])

result = model.predict(X)
result = result.argmax(1)

d = {'ImageId':range(1,X.shape[0]+1),'Label':result}
df = pd.DataFrame(data=d)
df.set_index('ImageId',inplace=True)
df.to_csv(args['submissionpath'])

'''
for img,res in zip(X,result):
	print(res)
	img = img.reshape(28,28)
	cv2.imshow('img',img)
	if cv2.waitKey(0) == ord('q'):
		break
'''






