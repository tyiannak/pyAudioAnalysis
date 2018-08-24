import sys, os, alsaaudio, time, audioop, numpy, glob,  scipy, subprocess, wave, cPickle, threading, shutil
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from scipy.fftpack import rfft
import audioFeatureExtraction as aF	
import audioTrainTest as aT
import audioSegmentation as aS
from scipy.fftpack import fft
import matplotlib
matplotlib.use('TkAgg')

Fs = 16000

def recordAudioSegments(RecordPath, BLOCKSIZE):	
	# This function is used for recording audio segments (until ctr+c is pressed)
	# ARGUMENTS:
	# - RecordPath:		the path where the wav segments will be stored
	# - BLOCKSIZE:		segment recording size (in seconds)
	# 
	# NOTE: filenames are based on clock() value
	
	print "Press Ctr+C to stop recording"
	RecordPath += os.sep
	d = os.path.dirname(RecordPath)
	if os.path.exists(d) and RecordPath!=".":
		shutil.rmtree(RecordPath)	
	os.makedirs(RecordPath)	

	inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE,alsaaudio.PCM_NONBLOCK)
	inp.setchannels(1)
	inp.setrate(Fs)
	inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
	inp.setperiodsize(512)
	midTermBufferSize = int(Fs*BLOCKSIZE)
	midTermBuffer = []
	curWindow = []
	elapsedTime = "%08.3f" % (time.time())
	while 1:
			l,data = inp.read()		   
		    	if l:
				for i in range(len(data)/2):
					curWindow.append(audioop.getsample(data, 2, i))
		
				if (len(curWindow)+len(midTermBuffer)>midTermBufferSize):
					samplesToCopyToMidBuffer = midTermBufferSize - len(midTermBuffer)
				else:
					samplesToCopyToMidBuffer = len(curWindow)

				midTermBuffer = midTermBuffer + curWindow[0:samplesToCopyToMidBuffer];
				del(curWindow[0:samplesToCopyToMidBuffer])
			

			if len(midTermBuffer) == midTermBufferSize:
				# allData = allData + midTermBuffer				
				curWavFileName = RecordPath + os.sep + str(elapsedTime) + ".wav"				
				midTermBufferArray = numpy.int16(midTermBuffer)
				wavfile.write(curWavFileName, Fs, midTermBufferArray)
				print "AUDIO  OUTPUT: Saved " + curWavFileName
				midTermBuffer = []
				elapsedTime = "%08.3f" % (time.time())	
	
def recordAnalyzeAudio(duration, outputWavFile, midTermBufferSizeSec, modelName, modelType):
	'''
	recordAnalyzeAudio(duration, outputWavFile, midTermBufferSizeSec, modelName, modelType)

	This function is used to record and analyze audio segments, in a fix window basis.

	ARGUMENTS: 
	- duration			total recording duration
	- outputWavFile			path of the output WAV file
	- midTermBufferSizeSec		(fix)segment length in seconds
	- modelName			classification model name
	- modelType			classification model type

	'''

	if modelType=='svm':
		[Classifier, MEAN, STD, classNames, mtWin, mtStep, stWin, stStep, compute_beat] = aT.load_model(modelName)
	elif modelType=='knn':
		[Classifier, MEAN, STD, classNames, mtWin, mtStep, stWin, stStep, compute_beat] = aT.load_model_knn(modelName)
	else:
		Classifier = None

	inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NONBLOCK)
	inp.setchannels(1)
	inp.setrate(Fs)
	inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
	inp.setperiodsize(512)
	midTermBufferSize = int(midTermBufferSizeSec * Fs)
	allData = []
	midTermBuffer = []
	curWindow = []
	count = 0

	while len(allData)<duration*Fs:
		# Read data from device
		l,data = inp.read()
	    	if l:
			for i in range(l):
				curWindow.append(audioop.getsample(data, 2, i))		
			if (len(curWindow)+len(midTermBuffer)>midTermBufferSize):
				samplesToCopyToMidBuffer = midTermBufferSize - len(midTermBuffer)
			else:
				samplesToCopyToMidBuffer = len(curWindow)
			midTermBuffer = midTermBuffer + curWindow[0:samplesToCopyToMidBuffer];
			del(curWindow[0:samplesToCopyToMidBuffer])
		if len(midTermBuffer) == midTermBufferSize:
			count += 1						
			if Classifier!=None:
				[mtFeatures, stFeatures, _] = aF.mtFeatureExtraction(midTermBuffer, Fs, 2.0*Fs, 2.0*Fs, 0.020*Fs, 0.020*Fs)
				curFV = (mtFeatures[:,0] - MEAN) / STD;
				[result, P] = aT.classifierWrapper(Classifier, modelType, curFV)
				print classNames[int(result)]
			allData = allData + midTermBuffer

			plt.clf()
			plt.plot(midTermBuffer)
			plt.show(block = False)
			plt.draw()


			midTermBuffer = []

	allDataArray = numpy.int16(allData)
	wavfile.write(outputWavFile, Fs, allDataArray)

def main(argv):
	if argv[1] == '-recordSegments':		# record input
		if (len(argv)==4): 			# record segments (until ctrl+c pressed)
			recordAudioSegments(argv[2], float(argv[3]))
		else:
			print "Error.\nSyntax: " + argv[0] + " -recordSegments <recordingPath> <segmentDuration>"

	if argv[1] == '-recordAndClassifySegments':	# record input
		if (len(argv)==6):			# recording + audio analysis
			duration = int(argv[2])
			outputWavFile = argv[3]
			modelName = argv[4]
			modelType = argv[5]
			if modelType not in ["svm", "knn"]:
				raise Exception("ModelType has to be either svm or knn!")
			if not os.path.isfile(modelName):
				raise Exception("Input modelName not found!")
			recordAnalyzeAudio(duration, outputWavFile, 2.0, modelName, modelType)
		else:
			print "Error.\nSyntax: " + argv[0] + " -recordAndClassifySegments <duration> <outputWafFile> <modelName> <modelType>"
	
if __name__ == '__main__':
	main(sys.argv)
