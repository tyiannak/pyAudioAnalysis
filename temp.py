import csv
import numpy as np
import os.path
import sklearn
import sklearn.hmm
import os
import glob
import cPickle
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
		transmat[int(labels[i]), int(labels[i+1])] += 1;
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

def trainHMM_fromFile(wavFile, gtFile, hmmModelName, mtWin, mtStep):
	[segStart, segEnd, segLabels] = readSegmentGT(gtFile)
	flags, classNames = segs2flags(segStart, segEnd, segLabels, mtStep)

	[Fs, x] = audioBasicIO.readAudioFile(wavFile);
	#F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.050*Fs);
	[F, _] = audioFeatureExtraction.mtFeatureExtraction(x, Fs, mtWin * Fs, mtStep * Fs, round(Fs*0.050), round(Fs*0.050));
	startprob, transmat, means, cov = trainHMM_computeStatistics(F, flags)

	hmm = sklearn.hmm.GaussianHMM(startprob.shape[0], "diag", startprob, transmat)
	hmm.means_ = means
	hmm.covars_ = cov

	fo = open(hmmModelName, "wb")
	cPickle.dump(hmm, fo, protocol = cPickle.HIGHEST_PROTOCOL)
	cPickle.dump(classNames,  fo, protocol = cPickle.HIGHEST_PROTOCOL)
	cPickle.dump(mtWin,  fo, protocol = cPickle.HIGHEST_PROTOCOL)
	cPickle.dump(mtStep,  fo, protocol = cPickle.HIGHEST_PROTOCOL)
    	fo.close()

	return hmm, classNames

def trainHMM_fromDir(dirPath, hmmModelName, mtWin, mtStep):
	flagsAll = np.array([])
	classesAll = []
	for i,f in enumerate(glob.glob(dirPath + os.sep + '*.wav')):
		wavFile = f;
		gtFile = f.replace('.wav', '.segments');
		[segStart, segEnd, segLabels] = readSegmentGT(gtFile)
		flags, classNames = segs2flags(segStart, segEnd, segLabels, mtStep)

		for c in classNames:
			if c not in classesAll:
				classesAll.append(c)

		[Fs, x] = audioBasicIO.readAudioFile(wavFile);
		[F, _] = audioFeatureExtraction.mtFeatureExtraction(x, Fs, mtWin * Fs, mtStep * Fs, round(Fs*0.050), round(Fs*0.050));

		lenF = F.shape[1]; lenL = len(flags); MIN = min(lenF, lenL)
		F = F[:, 0:MIN]	
		flags = flags[0:MIN]

		flagsNew = []
		for j, fl in enumerate(flags):
			flagsNew.append( classesAll.index( classNames[flags[j]] ) )

		flagsAll = np.append(flagsAll, np.array(flagsNew))

		if i==0:
			Fall = F;
		else:
			Fall = np.concatenate((Fall, F), axis = 1)

	startprob, transmat, means, cov = trainHMM_computeStatistics(Fall, flagsAll)
	hmm = sklearn.hmm.GaussianHMM(startprob.shape[0], "diag", startprob, transmat)
	hmm.means_ = means
	hmm.covars_ = cov

	fo = open(hmmModelName, "wb")
	cPickle.dump(hmm, fo, protocol = cPickle.HIGHEST_PROTOCOL)
	cPickle.dump(classesAll,  fo, protocol = cPickle.HIGHEST_PROTOCOL)
	cPickle.dump(mtWin,  fo, protocol = cPickle.HIGHEST_PROTOCOL)
	cPickle.dump(mtStep,  fo, protocol = cPickle.HIGHEST_PROTOCOL)
    	fo.close()

	return hmm, classesAll

def hmmSegmentation(wavFileName, hmmModelName, PLOT = False, gtFileName = ""):
	[Fs, x] = audioBasicIO.readAudioFile(wavFileName);					# read audio data

	try:
		fo = open(hmmModelName, "rb")
	except IOError:
       		print "didn't find file"
        	return
    	try:
		hmm     	= cPickle.load(fo)
		classesAll      = cPickle.load(fo)
		mtWin 		= cPickle.load(fo)
		mtStep 		= cPickle.load(fo)
    	except:
        	fo.close()
	fo.close()	

	#Features = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.050*Fs);	# feature extraction
	[Features, _] = audioFeatureExtraction.mtFeatureExtraction(x, Fs, mtWin * Fs, mtStep * Fs, round(Fs*0.050), round(Fs*0.050));
	labels = hmm.predict(Features.T)							# apply model
	if PLOT:										# plot results
		if os.path.isfile(gtFileName):
			[segStart, segEnd, segLabels] = readSegmentGT(gtFileName)		
			flagsGT, classNamesGT = segs2flags(segStart, segEnd, segLabels, mtStep)
			print classNamesGT
			flagsGTNew = []
			for j, fl in enumerate(flagsGT):
				if classNamesGT[flagsGT[j]] in classesAll:
					flagsGTNew.append( classesAll.index( classNamesGT[flagsGT[j]] ) )
				else:
					flagsGTNew.append( -1 )
			flagsGT = np.array(flagsGTNew)
			plt.plot(flagsGT+0.1,'r')	
		plt.plot(labels);
		plt.show()
	return labels

#trainHMM_fromFile("radio/train/small/Jazz_Line_up.wav", "radio/train/small/Jazz_Line_up.segments", "hmmTemp") 

#trainHMM_fromDir("radioFinal/train", "hmmTrain1", 1.0, 1.0)
#trainHMM_fromDir("radioFinal/train", "hmmTrain05", 1.0, 0.5)

hmmSegmentation("radioFinal/test/Annie_Nightningale_0_1450.wav", "hmmTrain05", PLOT = True, gtFileName = "radioFinal/test/Annie_Nightningale_0_1450.segments")
#hmmSegmentation("radioFinal/test/bbc2B.wav", "hmmTrain05", PLOT = True, gtFileName = "radioFinal/test/bbc2B.segments")
#hmmSegmentation("radioFinal/test/bbc3A.wav", "hmmTrain05", PLOT = True, gtFileName = "radioFinal/test/bbc3A.segments")
#hmmSegmentation("radioFinal/test/bbc51.wav", "hmmTrain", PLOT = True, gtFileName = "radioFinal/test/bbc51.segments")
#hmmSegmentation("radioFinal/test/redfm2.wav", "hmmTrain", PLOT = True, gtFileName = "radioFinal/test/redfm2.segments")
