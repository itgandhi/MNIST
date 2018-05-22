import numpy as np
import pandas as pd
from scipy.ndimage import interpolation
import cv2
import argparse
np.random.seed(1337)


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
	X = (df.values).reshape(df.values.shape[0],28,28)
	return(X)

def scale(X, eps = 0.001):
	return (X-np.min(X,axis=0))/(np.max(X,axis=0)+eps)

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
X = skew(X)
X = scale(X)

for index,img in enumerate(X):
	ret,thresh1 = cv2.threshold(img,0.65,1,cv2.THRESH_BINARY)
	X[index]=thresh1

X = X.reshape(X.shape[0],1,28,28)
print("shape X : ",X.shape)
#exit(0)

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





