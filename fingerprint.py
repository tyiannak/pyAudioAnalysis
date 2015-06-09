import sys, time, os, glob, numpy, mlpy, cPickle, aifc
from scipy.fftpack import fft
from numpy import NaN, Inf, arange, isscalar, array
import matplotlib.pyplot as plt
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import utilities
from pyAudioAnalysis import audioFeatureExtraction
from scipy.spatial import distance

def stFeatureExtractionFingerPrint(signal, Fs, Win, Step):
	"""
	"""

	nceps = 13
	Win = int(Win);
	Step = int(Step)
	# Signal normalization
	signal = numpy.double(signal)

	signal = signal / (2.0**15)
	DC     = signal.mean()	
	MAX    = (numpy.abs(signal)).max()
	signal = (signal - DC) / MAX

	N = len(signal)								# total number of samples
	curPos = 0
	countFrames = 0
	nFFT = Win/2
	
	[fbank, freqs] = audioFeatureExtraction.mfccInitFilterBanks(Fs, nFFT)				# compute the triangular filter banks used in the mfcc calculation	

	stFeatures = []

	while (curPos+Win-1<N):							# for each short-term window until the end of signal
		countFrames += 1
		x = signal[curPos:curPos+Win]					# get current window
		curPos = curPos + Step						# update window position
		X = abs(fft(x))							# get fft magnitude
		X = X[0:nFFT]							# normalize fft
		X = X / len(X)

		curMFCC = audioFeatureExtraction.stMFCC(X, fbank, nceps).copy()[0]
		#curMFCC = audioFeatureExtraction.stSpectralRollOff(X, 0.90, Fs)		
		stFeatures.append([curMFCC])
	stFeatures = numpy.squeeze(numpy.array(stFeatures))

	fFeature = numpy.zeros(stFeatures.shape)
	for i in range(1,stFeatures.shape[0]-1):
		if (stFeatures[i] > stFeatures[i-1]) and (stFeatures[i] > stFeatures[i+1]):
			fFeature[i] = 4;
		
		if (stFeatures[i] > stFeatures[i-1]) and (stFeatures[i] < stFeatures[i+1]):
			fFeature[i] = 3;
		
		if (stFeatures[i] < stFeatures[i-1]) and (stFeatures[i] > stFeatures[i+1]):
			fFeature[i] = 2;
		
		if (stFeatures[i] < stFeatures[i-1]) and (stFeatures[i] < stFeatures[i+1]):
			fFeature[i] = 1;	
	return fFeature


def generateDB(dirName):
	fNames = []
	features = []
	types = ('*.wav', )
	wavFilesList = []
	for files in types:
		wavFilesList.extend(glob.glob(os.path.join(dirName, files)))	
	wavFilesList = sorted(wavFilesList)
	
	for wavFile in wavFilesList:	
		[Fs, x] = audioBasicIO.readAudioFile(wavFile)
		F = stFeatureExtractionFingerPrint(x, Fs, 0.050*Fs, 0.050*Fs);	
		features.append(F)
		fNames.append(wavFile)
	return fNames, features

def similarityMatrix(F1, F2):
	S = (distance.cdist(F1, F2))	
	return S

fNames, features = generateDB('ads')

queryFile = "demoAd.wav"
[Fs, x] = audioBasicIO.readAudioFile(queryFile)
F = stFeatureExtractionFingerPrint(x, Fs, 0.050*Fs, 0.050*Fs);	


for i in range(len(features)):
	L, Path = mlpy.lcs_std(F, features[i])
	#dist, cost, path, = mlpy.dtw_subsequence(features[i], F)
	print L, Path[0].shape, Path[1].shape

'''
print fNames[3]
S = similarityMatrix(F, features[3]) 

print S.mean()

tol = 0.01
while (1):
	[Im, Jm] = numpy.unravel_index(S.argmin(), S.shape)	
	i = Im
	j = Jm	
	if S[i,j] >= tol:
		break;
	while S[i, j] < tol:
		S[i, j] = 10000000
		i += 1
		j += 1		
		if i>=S.shape[0] or j>=S.shape[1]:
			break;	
	
	#if i - Im > 20:
	#	print Im, Jm, i, j, S.shape, S[i-1,j-1]		
	#print S[Im, Jm]
	#print S[Im+1, Jm+1]

plt.imshow(S)
plt.show()
'''
