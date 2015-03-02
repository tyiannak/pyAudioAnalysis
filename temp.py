import csv
import numpy as np
import os.path
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
	return np.array(flags), classNames

def trainHMM_computeStatistics(features, labels):
	uLabels = np.unique(labels)
	nComps = len(uLabels)

	nFeatures = features.shape[0]

	print features.shape
	print labels.shape

	if features.shape[1] < labels.shape[0]:
		print "trainHMM warning: number of short-term feature vectors must be greater or equal to the labels length!"
		labels = labels[0:features.shape[1]]

	# compute prior probabilities:
	startprob = np.zeros((nComps,))
	for i,u in enumerate(uLabels):
		startprob[i] = np.count_nonzero(labels==u)
	startprob = startprob / startprob.sum()				# normalize prior probabilities

	# compute transition matrix:	
	transmat = np.zeros((nComps, nComps))
	for i in range(labels.shape[0]-1):
		transmat[labels[i], labels[i+1]] += 1;
	for i in range(nComps): 					# normalize rows of transition matrix:
		transmat[i, :] /= transmat[i, :].sum()

	means = np.zeros((nComps, nFeatures))
	for i in range(nComps):
		means[i,:] = np.matrix(features[:,np.nonzero(labels==uLabels[i])[0]].mean(axis=1))

	cov = np.zeros( (nComps, nFeatures) );
	for i in range(nComps):
		#cov[i,:,:] = np.cov(features[:,np.nonzero(labels==uLabels[i])[0]])		# for full cov!
		cov[i,:] = np.std(features[:,np.nonzero(labels==uLabels[i])[0]], axis = 1)

	return startprob, transmat, means, cov

def trainHMM_fromFile(wavFile, gtFile):
	[segStart, segEnd, segLabels] = readSegmentGT(gtFile)
	flags, classNames = segs2flags(segStart, segEnd, segLabels, 1.0)

	[Fs, x] = audioBasicIO.readAudioFile(wavFile);
	#F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.050*Fs);
	[F, _] = audioFeatureExtraction.mtFeatureExtraction(x, Fs, 1.0 * Fs, 1.0 * Fs, round(Fs*0.050), round(Fs*0.050));
	startprob, transmat, means, cov = trainHMM_computeStatistics(F, flags)

	hmm = sklearn.hmm.GaussianHMM(startprob.shape[0], "diag", startprob, transmat)
	hmm.means_ = means
	hmm.covars_ = cov
	return hmm

def hmmSegmentation(wavFileName, hmmModel, PLOT = False, gtFileName = ""):
	[Fs, x] = audioBasicIO.readAudioFile(wavFileName);					# read audio data
	#Features = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.050*Fs);	# feature extraction
	[Features, _] = audioFeatureExtraction.mtFeatureExtraction(x, Fs, 1.0 * Fs, 1.0 * Fs, round(Fs*0.050), round(Fs*0.050));
	labels = hmmModel.predict(Features.T)							# apply model
	if PLOT:										# plot results
		if os.path.isfile(gtFileName):
			[segStart, segEnd, segLabels] = readSegmentGT(gtFileName)
			flagsGT, classNamesGT = segs2flags(segStart, segEnd, segLabels, 1.0)
			plt.plot(flagsGT+0.1,'r')	
		plt.plot(labels);
		plt.show()
	return labels

#hmm = trainHMM_fromFile("data/count2.wav", "data/count2.segments")
#labels = hmmSegmentation("data/count.wav", hmm, True, "data/count.segments")
hmm = trainHMM_fromFile("bbc3A.wav", "bbc3A.csv")
labels = hmmSegmentation("bbc3C.wav", hmm, True, "bbc3C.csv")

