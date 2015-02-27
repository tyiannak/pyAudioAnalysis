import csv
import numpy as np
import sklearn
import sklearn.hmm
import matplotlib.pyplot as plt
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt


def readSegmentGT(gtFile):
	f  = open(gtFile, "rb")
	reader = csv.reader(f, delimiter=',')
	segStart = []; segEnd = []; segLabel = []
	for row in reader:	
		if len(row)==3:
			segStart.append(float(row[0]))
			segEnd.append(float(row[1]))
			segLabel.append((row[2]))
	return np.array(segStart), np.array(segEnd), segLabel

def segs2flags(segStart, segEnd, segLabel, winSize):
	flags = []
	classNames = list(set(segLabel))
	print classNames
	curPos = winSize / 2;
	while curPos < segEnd[-1]:
		for i in range(len(segStart)):
			if curPos > segStart[i] and curPos <=segEnd[i]:
				break;
		flags.append(classNames.index(segLabel[i]))
		#print curPos - segStart
		#print curPos - segEnd
		curPos += winSize
	return flags, classNames

def trainHMM(features, labels):
	labels = np.array(labels);
	uLabels = np.unique(labels)
	nComps = len(uLabels)

	nFeatures = features.shape[0]
	#sklearn.hmm.GaussianHMM._init(features, 'stmc')

#	startprob = np.zeros((nComps,1))
#	for i,u in enumerate(uLabels):
#		startprob[i] = np.nonzero(labels==u
	startprob = np.array([0.5, 0.5]);
	transmat = np.array( [ [0.5, 0.5], [0.5, 0.5] ]);
	hmm = sklearn.hmm.GaussianHMM(nComps, "diag", startprob, transmat)

	means = np.zeros((nComps, nFeatures))
	for i in range(nComps):
		means[i,:] = np.matrix(features[:,np.nonzero(labels==uLabels[i])[0]].mean(axis=1))

	cov = np.zeros( (nComps, nFeatures) );
	for i in range(nComps):
		#cov[i,:,:] = np.cov(features[:,np.nonzero(labels==uLabels[i])[0]])
		cov[i,:] = np.std(features[:,np.nonzero(labels==uLabels[i])[0]], axis = 1)
	print cov.shape
	hmm.means_ = means
	hmm.covars_ = cov

	return hmm

def applyHMM(hmm, features):
	labels = hmm.predict(features.T)
	#labels = hmm.decode(features.T)
	plt.plot(labels);
	plt.show()

[segStart, segEnd, segLabels] = readSegmentGT("data/count.segments")
flags, classNames = segs2flags(segStart, segEnd, segLabels, 0.05)
flags.append(0) # TODO


[Fs, x] = audioBasicIO.readAudioFile("data/count.wav");
F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.050*Fs);
[Fs, x2] = audioBasicIO.readAudioFile("data/count2.wav");
F2 = audioFeatureExtraction.stFeatureExtraction(x2, Fs, 0.050*Fs, 0.050*Fs);

hmm = trainHMM(F, flags)
#print hmm.means_.shape
#print hmm.covars_.shape
applyHMM(hmm, F2)
#applyHMM(hmm, F)



