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
	
	Y = df['label'].values	

	df=df.loc[:,df.columns != 'label']

	X = df.values
	return(X,Y)

def scale(X, eps = 0.001):
	return (X-np.min(X,axis=0))/(np.max(X,axis=0)+eps)

def nudge(X,Y):
	translations = [(0,-1),(0,1),(-1,0),(1,0)]
	data = []
	target = []

	for(image, label) in zip(X,Y):
		image = image.reshape(28,28)
	
		for(tX,tY) in translations:
			M = np.float32([[1,0,tX],[0,1,tY]])
			trans = cv2.warpAffine(image,M,(28,28))

			data.append(trans.flatten())
			target.append(label)
	
	return(np.array(data),np.array(target))


ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required = True, help = "path to dataset file")
ap.add_argument("-t","--test",required = True, type = float, help = "size of test split")
ap.add_argument("-s","--search", help = "whether or not a grid search should be performed", type = int, default = 0)
ap.add_argument("-m","--modelpath",required = True, help = "path to save trained model")

args = vars(ap.parse_args())


(X,Y) = load_data(args['dataset'])
X = X.astype("float32")
X = scale(X)

(trainX, testX, trainY, testY) = train_test_split(X,Y,test_size = args['test'], random_state = 42)	

(trainX,trainY) = nudge(trainX,trainY)

print("shape X : ",trainX.shape)
print("shape Y : ",trainY.shape)
#exit(0)


'''
for (img,label) in zip(trainX,trainY):
	print(label)
	cv2.imshow('img',img)
	if cv2.waitKey(0) == ord('q'):
		break
'''


if args["search"] == 1:
	print("SEARCHING LOGISTIC REGRESSION")
	params = {"C":[1.0,10.0,100.0]}
	start = time.time()
	gs = GridSearchCV(LogisticRegression(), params, n_jobs = -1, verbose = 1)
	gs.fit(trainX, trainY)

	print("done in %0.3fs" % (time.time() - start))
	print("best score: %0.3f"%(gs.best_score_))
	print("LOGISTIC REGRESSION PARAMETERS")
	bestParams = gs.best_estimator_.get_params()

	for p in sorted(params.keys()):
		print ("\t %s:%f" % (p,bestParams[p]))

	rbm = BernoulliRBM()
	logistic = LogisticRegression()
	classifier = Pipeline([("rbm",rbm),("logistic",logistic)])

	print("SEARCH RBM + LOGISTIC REGRESSION")

	params = {
		"rbm__learning_rate":[0.1,0.01,0.001],
		"rbm__n_iter":[20,40,80],
		"rbm__n_components":[50,100,200],
		"logistic__C":[1.0,10.0,100.0]}

	start = time.time()
	gs = GridSearchCV(classifier, params, n_jobs=-1, verbose = 1)
	gs.fit(trainX,trainY)

	print ("\ndone in %0.3fs" % (time.time() - start))
	print ("best score: %0.3f" % (gs.best_score_))
	print ("RBM + LOGISTIC REGRESSION PARAMETERS")
	bestParams = gs.best_estimator_.get_params()

	for p in sorted(params.keys()):
		print ("\t %s: %f" % (p, bestParams[p]))
 
	# show a reminder message
	print ("\nIMPORTANT")
	print ("Now that your parameters have been searched, manually set")
	print ("them and re-run this script with --search 0")

else:
	
	rbm = BernoulliRBM(n_components = 200, n_iter = 40, learning_rate = 0.01, verbose = True)
	logistic = LogisticRegression(C=1.0)

	classifier = Pipeline([("rbm",rbm),("logistic",logistic)])
	classifier.fit(trainX,trainY)	
	print("RBM + LOGISTIC REGRESSION ON ORIGINAL DATASET")
	print(classification_report(testY, classifier.predict(testX)))
	print("SAVING TRAINED MODEL..")
	joblib.dump(classifier,args["modelpath"])
