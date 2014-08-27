import sys, time, os, glob
import numpy
from numpy import NaN, Inf, arange, isscalar, array
import scipy.io.wavfile as wavfile
from scipy.fftpack import rfft
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct
from scipy.signal import fftconvolve
import mlpy
import cPickle
from matplotlib.mlab import find
import matplotlib.pyplot as plt
import scipy.io as sIO
import aifc
from scipy import linalg as la
import audioTrainTest as aT

eps = 0.00000001

def stZCR(frame):
	"""Computes zero crossing rate of frame"""
	count = len(frame)
	countZ = numpy.sum(numpy.abs(numpy.diff(numpy.sign(frame)))) / 2
	return (numpy.float64(countZ) / numpy.float64(count-1.0))

def stEnergy(frame):
	"""Computes signal energy of frame"""
	return numpy.sum(frame ** 2) / numpy.float64(len(frame))

def stEnergyEntropy(frame, numOfShortBlocks = 10):
	"""Computes entropy of energy"""
	Eol = numpy.sum(frame**2);	# total frame energy
	L = len(frame);
	subWinLength = numpy.floor(L / numOfShortBlocks);
	if L!=subWinLength* numOfShortBlocks:
    		frame = frame[0:subWinLength* numOfShortBlocks]
	# subWindows is of size [numOfShortBlocks x L]	
	subWindows = frame.reshape(subWinLength, numOfShortBlocks, order='F').copy()

	# compute normalized sub-frame energies:
	s = numpy.sum(subWindows**2, axis=0) / (Eol+eps) 			# NOTE: use axis=0 for numpy.sum with matrices, otherwise TOTAL MATRIX sum will be returned

	# compute entropy of the normalized sub-frame energies:
	Entropy = -numpy.sum(s*numpy.log2(s+eps))
	return Entropy
	
def stSpectralCentroidAndSpread(X, fs):	
	"""Computes spectral centroid of frame (given abs(FFT))"""
	ind = (numpy.arange(1, len(X)+1)) * (fs/(2.0*len(X)))

	Xt = X.copy()
	Xt = Xt / Xt.max()
	NUM = numpy.sum(ind * Xt)
	DEN = numpy.sum(Xt) + eps
	C = (NUM/DEN)

	S = numpy.sqrt(numpy.sum(((ind-C)**2)*Xt)/ DEN);

	C = C / (fs/2.0);
	S = S / (fs/2.0);

	return (C, S)

def stSpectralEntropy(X, numOfShortBlocks = 10):
	"""Computes the spectral entropy"""

	# number of frame samples:
	L = len(X);

	# total spectral energy 
	Eol = numpy.sum(X**2);

	# length of sub-frame:
	subWinLength = numpy.floor(L / numOfShortBlocks);
	if L!=subWinLength* numOfShortBlocks:
		X = X[0:subWinLength* numOfShortBlocks];

	# define sub-frames (using matrix reshape):
	subWindows = X.reshape(subWinLength, numOfShortBlocks, order='F').copy()

	# compute spectral sub-energies:
	s = numpy.sum(subWindows**2, axis = 0) / (Eol+eps);

	# compute spectral entropy:
	En = -numpy.sum(s*numpy.log2(s+eps));

	return En

def stSpectralFlux(X, Xprev):
	"""
	Computes the spectral flux feature of the current frame
	ARGUMENTS:
		X:		the abs(fft) of the current frame
		Xpre:		the abs(fft) of the previous frame
	"""
	# compute the spectral flux as the sum of square distances:
	sumX = numpy.sum(X+eps)
	sumPrevX = numpy.sum(Xprev+eps)
	F = numpy.sum((X/sumX - Xprev/sumPrevX)**2);

	return F


def stSpectralRollOff(X, c, fs):	
	"""Computes spectral roll-off"""
	totalEnergy = numpy.sum(X**2);
	fftLength = len(X);	
	Thres = c*totalEnergy
	# find the spectral rolloff as the frequency position where the respective spectral energy is equal to c*totalEnergy
	CumSum = numpy.cumsum(X**2)+eps
	[a,] = numpy.nonzero(CumSum>Thres)
	mC = numpy.float64(a[0])/(float(fftLength));
	return (mC)

def stHarmonic(frame, fs):
	"""
	Computes harmonic ratio and pitch
	"""
	M = numpy.round(0.016 * fs) - 1
	R = numpy.correlate(frame, frame, mode='full')

	g = R[len(frame)-1]
	R = R[len(frame):-1]

	# estimate m0 (as the first zero crossing of R)
	[a,] = numpy.nonzero(numpy.diff(numpy.sign(R)))

	if len(a)==0:
		m0 = len(R)-1
	else:
		m0 = a[0]
	if M > len(R):
		M = len(R)-1

	Gamma = numpy.zeros((M), dtype=numpy.float64);
	CSum = numpy.cumsum(frame**2);
	Gamma[m0:M] = R[m0:M] / (numpy.sqrt((g*CSum[M:m0:-1]))+eps);

	ZCR = stZCR(Gamma)

	if ZCR>0.15:
		HR = 0.0
		f0 = 0.0
	else:
		if len(Gamma)==0:
			HR = 1.0
			blag = 0.0;
			Gamma = numpy.zeros((M), dtype=numpy.float64);
		else:
			HR = numpy.max(Gamma)
			blag = numpy.argmax(Gamma)

	    	# get fundamental frequency:
		f0 = fs / (blag+eps)
		if f0>5000:
			f0 = 0.0
		if HR<0.1:
			f0 = 0.0
		
#	print HR, f0
	return (HR, f0)

def mfcc_tfb(fs, nfft, lowfreq, linsc, logsc, nlinfilt, nlogfilt):
    """Compute triangular filterbank for MFCC computation."""
    # Total number of filters
    nfilt = nlinfilt + nlogfilt

    # Compute frequency points of the triangle:
    freqs = numpy.zeros(nfilt+2)
    freqs[:nlinfilt] = lowfreq + numpy.arange(nlinfilt) * linsc
    freqs[nlinfilt:] = freqs[nlinfilt-1] * logsc ** numpy.arange(1, nlogfilt + 3)
    heights = 2./(freqs[2:] - freqs[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = numpy.zeros((nfilt, nfft))
    # FFT bins (in Hz)
    nfreqs = numpy.arange(nfft) / (1. * nfft) * fs
    
    for i in range(nfilt):
        low = freqs[i]
        cen = freqs[i+1]
        hi = freqs[i+2]

        lid = numpy.arange(numpy.floor(low * nfft / fs) + 1,
                        numpy.floor(cen * nfft / fs) + 1, dtype=numpy.int)
        lslope = heights[i] / (cen - low)
        rid = numpy.arange(numpy.floor(cen * nfft / fs) + 1,
                        numpy.floor(hi * nfft / fs) + 1, dtype=numpy.int)
        rslope = heights[i] / (hi - cen)
        fbank[i][lid] = lslope * (nfreqs[lid] - low)
        fbank[i][rid] = rslope * (hi - nfreqs[rid])

    return fbank, freqs

def stMFCC(X, fbank, nceps):
	"""
	compute MFCCs
	
	ARGUMENTS:
		X:	abs(FFT)
		fbank:	filter bank (see mfcc_tfb)
	RETURN
		ceps:	MFCCs (13)
	"""
	mspec = numpy.log10(numpy.dot(X, fbank.T)+eps)
	ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:nceps]
	return ceps

def stSpectogram(signal, Fs, Win, Step, PLOT=False):
	"""
	Short-term FFT mag for spectogram estimation:
	Returns:
		a numpy array (nFFT x numOfShortTermWindows)
	ARGUMENTS:
		signal:		the input signal samples
		Fs:		the sampling freq (in Hz)
		Win:		the short-term window size (in samples)
		Step:		the short-term window step (in samples)
		PLOT:		flag, 1 if results are to be ploted
	RETURNS:
	"""

	signal = numpy.double(signal)
	signal = signal / (2.0**15)
	DC     = signal.mean()
        MAX    = (numpy.abs(signal)).max()
	signal = (signal - DC) / (MAX - DC)

	N = len(signal)		# total number of signals
	curPos = 0
	countFrames = 0
	nfft = Win / 2
	specgram = numpy.array([], dtype=numpy.float64)

	while (curPos+Win-1<N):
		countFrames += 1
		x = signal[curPos:curPos+Win]
		curPos = curPos + Step		
		X = abs(fft(x))
		X = X[0:nfft]
		X = X / len(X)

		if countFrames==1:
			specgram = X**2
		else:
			specgram = numpy.vstack((specgram, X))

	FreqAxis = [((f+1) * Fs) / (2*nfft)  for f in range(specgram.shape[1])]
	TimeAxis = [(t * Step) / Fs for t in range(specgram.shape[0])]

	if (PLOT):	
		fig, ax = plt.subplots()
		imgplot = plt.imshow(specgram.transpose()[ ::-1,:])
		Fstep = int(nfft / 5.0)
		FreqTicks = range(0, int(nfft) + Fstep, Fstep)
		FreqTicksLabels = [str(Fs/2-int((f*Fs) / (2*nfft))) for f in FreqTicks]
		ax.set_yticks(FreqTicks)
		ax.set_yticklabels(FreqTicksLabels)
		TStep = countFrames/3
		TimeTicks = range(0, countFrames, TStep)
		TimeTicksLabels = ['%.2f' % (float(t * Step) / Fs) for t in TimeTicks]
		ax.set_xticks(TimeTicks)
		ax.set_xticklabels(TimeTicksLabels)
		ax.set_xlabel('time (secs)')
		ax.set_ylabel('freq (Hz)')
		imgplot.set_cmap('jet')
		plt.colorbar()
		plt.show()

	return (specgram, TimeAxis, FreqAxis)

def stFeatureExtraction(signal, Fs, Win, Step):
	"""
	Short-term feature extraction:
	Returns:
		a numpy array (numOfFeatures x numOfShortTermWindows)
	ARGUMENTS
		signal:		the input signal samples
		Fs:		the sampling freq (in Hz)
		Win:		the short-term window size (in samples)
		Step:		the short-term window step (in samples)
	RETURNS
		stFeatures
	"""

	signal = numpy.double(signal)
	signal = signal / (2.0**15)
	DC     = signal.mean()	
        MAX    = (numpy.abs(signal)).max()
	signal = (signal - DC) / MAX
	# print (numpy.abs(signal)).max()

	N = len(signal)		# total number of signals
	curPos = 0
	countFrames = 0
   	# MFCC parameters: taken from auditory toolbox

	# MFCCs taken from http://pydoc.net/Python/scikits.talkbox/0.2.5/scikits.talkbox.features.mfcc/ (NOT USED DIRECTLY FROM THERE... only DCT HAS BEEN USED AS IT IS):
    	lowfreq = 133.33; linsc = 200/3.; logsc = 1.0711703; nlinfil = 13; nlogfil = 27; nceps = 13; nfil = nlinfil + nlogfil; nfft = Win / 2
	if Fs < 8000:
		nlogfil = 5
		nfil = nlinfil + nlogfil; nfft = Win / 2

	# compute filter banks for mfcc:
	[fbank, freqs] = mfcc_tfb(Fs, nfft, lowfreq, linsc, logsc, nlinfil, nlogfil)

	numOfTimeSpectralFeatures = 8
	numOfHarmonicFeatures = 0
	totalNumOfFeatures = numOfTimeSpectralFeatures + nceps + numOfHarmonicFeatures
	stFeatures = numpy.array([], dtype=numpy.float64)

	while (curPos+Win-1<N):
		countFrames += 1
		x = signal[curPos:curPos+Win]
		curPos = curPos + Step		
		X = abs(fft(x))
		X = X[0:nfft]
		X = X / len(X)
		if countFrames==1:
			Xprev = X.copy()
		curFV = numpy.zeros((totalNumOfFeatures, 1))
		curFV[0] = stZCR(x)									# zero crossing rate
		curFV[1] = stEnergy(x)									# short-term energy
		curFV[2] = stEnergyEntropy(x)								# short-term entropy of energy
		[curFV[3], curFV[4]] = stSpectralCentroidAndSpread(X, Fs)				# spectral centroid and spread
		curFV[5] = stSpectralEntropy(X)								# spectral entropy
		curFV[6] = stSpectralFlux(X, Xprev)							# spectral flux
		curFV[7] = stSpectralRollOff(X, 0.90, Fs)						# spectral rolloff
		curFV[numOfTimeSpectralFeatures:numOfTimeSpectralFeatures+nceps,0] = stMFCC(X, fbank, nceps).copy()
		#HR, curFV[numOfTimeSpectralFeatures+nceps] = stHarmonic(x, Fs)
		
		# curFV[numOfTimeSpectralFeatures+nceps+1] = freq_from_autocorr(x, Fs)
		# print curFV[numOfTimeSpectralFeatures+nceps], curFV[numOfTimeSpectralFeatures+nceps+1]
		if countFrames==1:
			stFeatures = curFV
		else:
			stFeatures = numpy.concatenate((stFeatures, curFV), 1)
		Xprev = X.copy()
	
#		if curFV[numOfTimeSpectralFeatures+nceps+1]>0:
#			print curFV[numOfTimeSpectralFeatures+nceps], curFV[numOfTimeSpectralFeatures+nceps+1]

	return numpy.array(stFeatures)

def stFeatureSpeed(signal, Fs, Win, Step):

	signal = numpy.double(signal)
	signal = signal / (2.0**15)
	DC     = signal.mean()	
        MAX    = (numpy.abs(signal)).max()
	signal = (signal - DC) / MAX
	# print (numpy.abs(signal)).max()

	N = len(signal)		# total number of signals
	curPos = 0
	countFrames = 0
   	# MFCC parameters: taken from auditory toolbox

	# MFCCs taken from http://pydoc.net/Python/scikits.talkbox/0.2.5/scikits.talkbox.features.mfcc/ (NOT USED DIRECTLY FROM THERE... only DCT HAS BEEN USED AS IT IS):
    	lowfreq = 133.33; linsc = 200/3.; logsc = 1.0711703; nlinfil = 13; nlogfil = 27; nceps = 13; nfil = nlinfil + nlogfil; nfft = Win / 2
	if Fs < 8000:
		nlogfil = 5
		nfil = nlinfil + nlogfil; nfft = Win / 2

	# compute filter banks for mfcc:
	[fbank, freqs] = mfcc_tfb(Fs, nfft, lowfreq, linsc, logsc, nlinfil, nlogfil)

	numOfTimeSpectralFeatures = 8
	numOfHarmonicFeatures = 1
	totalNumOfFeatures = numOfTimeSpectralFeatures + nceps + numOfHarmonicFeatures
	#stFeatures = numpy.array([], dtype=numpy.float64)
	stFeatures = [];

	while (curPos+Win-1<N):
		countFrames += 1
		x = signal[curPos:curPos+Win]
		curPos = curPos + Step		
		X = abs(fft(x))
		X = X[0:nfft]
		X = X / len(X)
		Ex = 0.0
		El = 0.0
		X[0:4] = 0;
#		M = numpy.round(0.016 * fs) - 1
#		R = numpy.correlate(frame, frame, mode='full')
		stFeatures.append(stHarmonic(x, Fs))
#		for i in range(len(X)):
			#if (i < (len(X) / 8)) and (i > (len(X)/40)):
			#	Ex += X[i]*X[i]
			#El += X[i]*X[i]

#		stFeatures.append(Ex / El)
#		stFeatures.append(numpy.argmax(X))

		
#		if curFV[numOfTimeSpectralFeatures+nceps+1]>0:
#			print curFV[numOfTimeSpectralFeatures+nceps], curFV[numOfTimeSpectralFeatures+nceps+1]

	return numpy.array(stFeatures)

def mtFeatureExtraction(signal, Fs, mtWin, mtStep, stWin, stStep):
	"""
	Mid-term feature extraction
	"""

	mtWinRatio  = round(mtWin  / stStep);
	mtStepRatio = round(mtStep / stStep);


	mtFeatures = []
	
	stFeatures = stFeatureExtraction(signal, Fs, stWin, stStep)
	numOfFeatures = len(stFeatures)
	numOfStatistics = 2;	

	mtFeatures = []
	#for i in range(numOfStatistics * numOfFeatures + 1):
	for i in range(numOfStatistics * numOfFeatures):
		mtFeatures.append([])

	for i in range(numOfFeatures):		# for each of the short-term features:
		curPos = 0
		N = len(stFeatures[i])
		while (curPos<N):
			N1 = curPos
			N2 = curPos + mtWinRatio
			if N2 > N:
				N2 = N
			curStFeatures = stFeatures[i][N1:N2]

			mtFeatures[i].append(numpy.mean(curStFeatures))
			mtFeatures[i+numOfFeatures].append(numpy.std(curStFeatures))
			#mtFeatures[i+2*numOfFeatures].append(numpy.std(curStFeatures) / (numpy.mean(curStFeatures)+0.00000010))
			curPos += mtStepRatio
	#mtFeatures[-1].append(len(signal) / float(Fs))		

	return numpy.array(mtFeatures), stFeatures

def readAudioFile(path):
	extension = os.path.splitext(path)[1]

	try:
		if extension.lower() == '.wav':
			[Fs, x] = wavfile.read(path)
		elif extension.lower() == '.aif' or extension.lower() == '.aiff':
			s = aifc.open(path, 'r')
			nframes = s.getnframes()
			strsig = s.readframes(nframes)
			x = numpy.fromstring(strsig, numpy.short).byteswap()
			Fs = s.getframerate()
		else:
			print "Error in readAudioFile(): Unknown file type!"
			return (-1,-1)
	except IOError:	
		print "Error: file not found or other I/O error."
		return (-1,-1)
	return (Fs, x)

def dirWavFeatureExtraction(dirName, mtWin, mtStep, stWin, stStep):
	"""
	This function extracts the mid-term features of the WAVE files of a particular folder.
	NOTE: The resulting feature vector is extracted by long-term averaging the mid-term features. 
	
	ARGUMENTS:
		- dirName:		the path of the WAVE directory
		- mtWin, mtStep:	mid-term window and step (in seconds)
		- stWin, stStep:	short-term window and step (in seconds)
	"""

	allMtFeatures = numpy.array([])
 	processingTimes = []

	types = ('*.wav', '*.aif',  '*.aiff')
	wavFilesList = []
	for files in types:
		wavFilesList.extend(glob.glob(os.path.join(dirName, files)))
	
	wavFilesList = sorted(wavFilesList)
	
	for wavFile in wavFilesList:	
		[Fs, x] = readAudioFile(wavFile)			# read file
		t1 = time.clock()
		x = stereo2mono(x);					# convert stereo to mono
		[MidTermFeatures, _] = 	mtFeatureExtraction(x, Fs, round(mtWin*Fs), round(mtStep*Fs), round(Fs*stWin), round(Fs*stStep))
									# mid-term feature

		MidTermFeatures = numpy.transpose(MidTermFeatures)
		MidTermFeatures = MidTermFeatures.mean(axis=0)		# long term averaging of mid-term statistics
		if len(allMtFeatures)==0:				# append feature vector
			allMtFeatures = MidTermFeatures
		else:
			allMtFeatures = numpy.vstack((allMtFeatures, MidTermFeatures))
		t2 = time.clock()
		duration = float(len(x)) / Fs
		processingTimes.append((t2-t1) / duration)
	if len(processingTimes)>0:
		print "Feature extraction complexity ratio: {0:.1f} x realtime".format((1.0/numpy.mean(numpy.array(processingTimes))))
	return (allMtFeatures, wavFilesList)

def dirsWavFeatureExtraction(dirNames, mtWin, mtStep, stWin, stStep):
	'''
	Same as dirWavFeatureExtraction, but instead of a single dir it takes a list of paths as input and returns a list of feature matrices.
	EXAMPLE:
	[features, classNames] = a.dirsWavFeatureExtraction(['audioData/classSegmentsRec/noise','audioData/classSegmentsRec/speech','audioData/classSegmentsRec/brush-teeth','audioData/classSegmentsRec/shower'], 1, 1, 0.02, 0.02);
	'''
	# feature extraction for each class:
	features = [];
	classNames = []
	fileNames = []
	for i,d in enumerate(dirNames):
		[f, fn] = dirWavFeatureExtraction(d, mtWin, mtStep, stWin, stStep)
		if f.shape[0] > 0: # if at least one audio file has been found in the provided folder:
			features.append(f)
			fileNames.append(fn)
			if d[-1] == "/":
				classNames.append(d.split(os.sep)[-2])
			else:
				classNames.append(d.split(os.sep)[-1])
	return features, classNames, fileNames

def stereo2mono(x):
	if x.ndim==1:
		return x
	else:
		if x.ndim==2:
			return ( (x[:,1] / 2) + (x[:,0] / 2) )
		else:
			return -1

def mtFeatureExtractionToFile(fileName, midTermSize, midTermStep, shortTermSize, shortTermStep, outFile):
	[Fs, x] = readAudioFile(fileName)
	x = stereo2mono(x);
	[MidTermFeatures, _] = mtFeatureExtraction(x, Fs, round(Fs*midTermSize), round(Fs*midTermStep), round(Fs*shortTermSize), round(Fs*shortTermStep))
	numpy.save(outFile, MidTermFeatures)

def mtFeatureExtractionToFileDir(dirName, midTermSize, midTermStep, shortTermSize, shortTermStep):
	types = (dirName+os.sep+'*.wav',) # the tuple of file types
	filesToProcess = []
	for files in types:
		filesToProcess.extend(glob.glob(files))
	for f in filesToProcess:
		mtFeatureExtractionToFile(f, midTermSize, midTermStep, shortTermSize, shortTermStep, f)

def main(argv):
	if (len(argv)==7):
		# single audio file feature extraction
		wavFileName = argv[1]
		midTermSize = float(argv[2])
		midTermStep = float(argv[3])
		shortTermSize = float(argv[4])
		shortTermStep = float(argv[5])
		outFile = str(argv[6])
		startTime = time.clock()
		mtFeatureExtractionToFile(wavFileName, midTermSize, midTermStep, shortTermSize, shortTermStep, outFile)
		endTime = time.clock()
		print endTime-startTime

if __name__ == '__main__':
	main(sys.argv)
