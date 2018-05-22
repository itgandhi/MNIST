
import numpy as np
from scipy.ndimage import interpolation
import pandas as pd
import cv2
import argparse
np.random.seed(1337)

from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
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
			ret,thresh1 = cv2.threshold(image,0.65,1,cv2.THRESH_BINARY)
			data.append(cv2.warpAffine(thresh1,M,(28,28)).reshape(1,28,28))
			target.append(label)
	
	return(np.array(data),np.array(target))


def moments(image):
    c0,c1 = np.mgrid[:image.shape[0],:image.shape[1]] # A trick in numPy to create a mesh grid
    totalImage = np.sum(image) #sum of pixels
    m0 = np.sum(c0*image)/totalImage #mu_x
    m1 = np.sum(c1*image)/totalImage #mu_y
    m00 = np.sum((c0-m0)**2*image)/totalImage #var(x)
    m11 = np.sum((c1-m1)**2*image)/totalImage #var(y)
    m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage #covariance(x,y)
    mu_vector = np.array([m0,m1]) # Notice that these are \mu_x, \mu_y respectively
    covariance_matrix = np.array([[m00,m01],[m01,m11]]) # Do you see a similarity between the covariance matrix
    return mu_vector, covariance_matrix

def skew(X):
	print('SKEWING...')
	for index,img in enumerate(X):
		c,v = moments(img)
		alpha = v[0,1]/v[0,0]
		affine = np.array([[1,0],[alpha,1]])
		ocenter = np.array(img.shape)/2.0
		offset = c-np.dot(affine,ocenter)
		tempimg = interpolation.affine_transform(img,affine,offset=offset)
		X[index] = (tempimg - tempimg.min()) / (tempimg.max() - tempimg.min())

		'''
		cv2.imshow('img',img)
		cv2.imshow('skew', rotated)
		cv2.imshow('skew1', rotated1)
		if cv2.waitKey(0) == ord('q'):
			break
		'''
	print('SKEW COMPLETE...')
	return X


def larger_model(modelpath = None):

	
	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	
	if modelpath is not None:	
		model.load_weights(modelpath)

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required = True, help = "path to dataset file")
ap.add_argument("-t","--test",required = True, type = float, help = "size of test split. eg. 0.4 = 60% train 40% test...")

ap.add_argument("-m","--modelpath",required = True, help = "path to save trained model")
ap.add_argument("-tm","--trainedModelPath",required = False, help = "path to save trained model")

args = vars(ap.parse_args())

(X,Y) = load_data(args['dataset'])
X = X.astype("float32")

(X_train, y_train),(X_test,y_test) = mnist.load_data()
X_train = np.concatenate((X_train,X_test),axis = 0)
y_train = np.concatenate((y_train,y_test),axis = 0)

X = np.concatenate((X,X_train),axis = 0)
Y = np.concatenate((Y,y_train),axis = 0)

print("shape X : ",X.shape)
print("shape Y : ",Y.shape)

X = skew(X)

X = scale(X)
(X,Y) = nudge(X,Y)

(trainX, testX, trainY, testY) = train_test_split(X,Y,test_size = args['test'], random_state = 42)

del(X)
del(Y)

'''
for img,label in zip(trainX,trainY):
	img = img.reshape(28,28)
	print(label,np.max(img),np.min(img))
	
	cv2.imshow('img', img)
	
	
	if cv2.waitKey(0) == ord('q'):
		break
'''

trainY = np_utils.to_categorical(trainY)
testY = np_utils.to_categorical(testY)
num_classes = testY.shape[1]

if(args['trainedModelPath']):
	model = larger_model(args['trainedModelPath'])
else:
	model = larger_model()

checkpoint = ModelCheckpoint(args['modelpath']+'/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
earlystopper = EarlyStopping(monitor='val_acc', patience=15, verbose=1, mode='max')
model.fit(trainX, trainY, validation_data=(testX,testY), epochs=1000, batch_size=200,verbose=1, callbacks=[earlystopper,checkpoint])
#model.fit(trainX, trainY, validation_data=(testX,testY), epochs=1000, batch_size=200,verbose=1, callbacks=[checkpoint])

scores = model.evaluate(testX,testY, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))

