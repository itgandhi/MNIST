from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import argparse
import time
import cv2

def load_data(dataPath):
	df = pd.read_csv(dataPath)
	
	df=df.loc[:,df.columns != 'label']

	X = df.values
	return(X)

def scale(X, eps = 0.001):
	return (X-np.min(X,axis=0))/(np.max(X,axis=0)+eps)

ap = argparse.ArgumentParser()
ap.add_argument("-t","--testfile",required = True, help = "path to test file")

ap.add_argument("-m","--modelpath",required = True , help = "path model..", default = 0)

ap.add_argument("-s","--submissionpath",required = True , help = "path where you want to store sbmission file..", default = 0)
args = vars(ap.parse_args())

X = load_data(args['testfile'])
X = X.astype("float32")
X = scale(X)

print("shape X : ",X.shape)

clf = joblib.load(args['modelpath'])

predicted = clf.predict(X)

d = {'ImageId':range(1,X.shape[0]+1),'Label':predicted}
df = pd.DataFrame(data=d)
df.set_index('ImageId',inplace=True)
#print(df)

#df.to_csv(args['submissionpath'])


for index,img in enumerate(X):
	print(predicted[index])
	img = img.reshape(28,28)
	cv2.imshow('img',img)
	if cv2.waitKey(0) == ord('q'):
		break



