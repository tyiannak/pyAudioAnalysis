#from pyAudioAnalysis import audioSegmentation as aS
#[flagsInd, classesAll, acc] = aS.mtFileClassification("data/scottish.wav", "data/svmSM", "svm", True, 'data/scottish.segments')

import os, numpy, mlpy
import matplotlib.pyplot as plt
from pyAudioAnalysis import audioSegmentation as aS
from sklearn.lda import LDA
import audioFeatureExtraction as aF
import audioTrainTest as aT
#aS.trainHMM_fromFile('radioFinal/train/bbc4A.wav', 'radioFinal/train/bbc4A.segments', 'hmmTemp1', 1.0, 1.0)	# train using a single file
#aS.trainHMM_fromDir('radioFinal/small/', 'hmmTemp2', 1.0, 1.0)							# train using a set of files in a folder
#aS.hmmSegmentation('data/scottish.wav', 'hmmTemp1', True, 'data/scottish.segments')				# test 1
#aS.hmmSegmentation('data/scottish.wav', 'hmmTemp2', True, 'data/scottish.segments')				# test 2


dirName = "/home/tyiannak/Desktop/DIARIZATION_ALL/train"
dirName2 = "/home/tyiannak/Desktop/DIARIZATION_ALL/test"
listOfDirs  = [ os.path.join(dirName, name) for name in os.listdir(dirName) if os.path.isdir(os.path.join(dirName, name)) ]
listOfDirs2 = [ os.path.join(dirName2, name) for name in os.listdir(dirName2) if os.path.isdir(os.path.join(dirName2, name)) ]


mtWin = 2.0; 
mtStep = 2.0;
stWin = 0.050;
stStep = 0.050;

aT.featureAndTrain(listOfDirs, mtWin, mtStep, stWin, stStep, "knn", "knnSpeaker", computeBEAT = False, perTrain = 0.50)
[Classifier, MEAN, STD, classNames, mtWin, mtStep, stWin, stStep, computeBEAT] = aT.loadKNNModel("knnSpeaker")
[features, classNames, _]   = aF.dirsWavFeatureExtraction(listOfDirs2, mtWin, mtStep, stWin, stStep, computeBEAT = False)
[X, Y] = aT.listOfFeatures2Matrix(features)
featuresP = [];
for i in range(X.shape[0]):
	X[i,:] = (X[i,:] - MEAN) / STD
	[Result, P] = aT.classifierWrapper(Classifier, "knn", X[i,:])	
	print i, P
	featuresP.append(P)
featuresP = numpy.matrix(featuresP)
print featuresP.shape
cls, means, steps = mlpy.kmeans(featuresP, k=6, plus=True)
plt.plot(cls)
plt.plot(Y,'r')
plt.show()

#(feature2, MEAN, STD) = aT.normalizeFeatures(features)
#classifierParams = numpy.array([1, 3])
#bestParam = aT.evaluateClassifier(features, classNames, 100, "knn", classifierParams, 0, 0.50)

#[features2, classNames2, _] = aF.dirsWavFeatureExtraction(listOfDirs2, mtWin, mtStep, stWin, stStep, computeBEAT = False)

#[X, Y] = aT.listOfFeatures2Matrix(features)
#(X, MEAN, STD) = aT.normalizeFeatures([X])
#X = X[0]
#print MEAN.shape
#clf = LDA(n_components=6)
#clf.fit(X, Y)
#[X2, Y2] = aT.listOfFeatures2Matrix(features2)
#for i in range(X2.shape[0]):
#	X2[i,:] = (X2[i,:] - MEAN) / STD
#X2new =  clf.transform(X2)
#cls, means, steps = mlpy.kmeans(X2new, k=6, plus=True)
#print Y2.shape, X2new.shape
#print Y2
#plt.plot(cls)
#plt.plot(Y2,'--c')
#plt.show()
#plt.plot(Y2)
#plt.show()


