import os, sys, shutil, glob, numpy
import scipy.io.wavfile as wavfile
import audioBasicIO
import audioTrainTest as aT
import audioSegmentation as aS
import matplotlib.pyplot as plt

minDuration = 7;

def classifyFolderWrapper(inputFolder, modelType, modelName, outputMode=False):
	if not os.path.isfile(modelName):
		raise Exception("Input modelName not found!")

	if modelType=='svm':
		[Classifier, MEAN, STD, classNames, mtWin, mtStep, stWin, stStep, computeBEAT] = aT.loadSVModel(modelName)
	elif modelType=='knn':
		[Classifier, MEAN, STD, classNames, mtWin, mtStep, stWin, stStep, computeBEAT] = aT.loadKNNModel(modelName)

	PsAll = numpy.zeros((len(classNames), ))	
		
	files = "*.wav"
	if os.path.isdir(inputFolder):
		strFilePattern = os.path.join(inputFolder, files)
	else:
		strFilePattern = inputFolder + files

	wavFilesList = []
	wavFilesList.extend(glob.glob(strFilePattern))
	wavFilesList = sorted(wavFilesList)
	if len(wavFilesList)==0:
		print "No WAV files found!"
		return 
	
	Results = []
	for wavFile in wavFilesList:	
		[Fs, x] = audioBasicIO.readAudioFile(wavFile)	
		signalLength = x.shape[0] / float(Fs)
		[Result, P, classNames] = aT.fileClassification(wavFile, modelName, modelType)					
		PsAll += (numpy.array(P) * signalLength)		
		Result = int(Result)
		Results.append(Result)
		if outputMode:
			print "{0:s}\t{1:s}".format(wavFile,classNames[Result])
	Results = numpy.array(Results)
	
	# print distribution of classes:
	[Histogram, _] = numpy.histogram(Results, bins=numpy.arange(len(classNames)+1))
	for i,h in enumerate(Histogram):
		print "{0:20s}\t\t{1:d}".format(classNames[i], h)
	print PsAll
	print classNames
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.title("Classes percentage " + inputFolder.replace('Segments',''))
	ax.axis((0, len(classNames)+1, 0, 100))
	ax.set_xticks(numpy.array(range(len(classNames)+1)))
	ax.set_xticklabels([" "] + classNames)
	ax.bar(numpy.array(range(len(classNames)))+0.5, PsAll)
	plt.show()

def getMusicSegmentsFromFile(inputFile):	
	modelType = "svm"
	modelName = "data/svmMovies8classes"
	
	dirOutput = inputFile[0:-4] + "_musicSegments"
	
	if os.path.exists(dirOutput) and dirOutput!=".":
		shutil.rmtree(dirOutput)	
	os.makedirs(dirOutput)	
	
	[Fs, x] = audioBasicIO.readAudioFile(inputFile)	

	if modelType=='svm':
		[Classifier, MEAN, STD, classNames, mtWin, mtStep, stWin, stStep, computeBEAT] = aT.loadSVModel(modelName)
	elif modelType=='knn':
		[Classifier, MEAN, STD, classNames, mtWin, mtStep, stWin, stStep, computeBEAT] = aT.loadKNNModel(modelName)

	flagsInd, classNames, acc = aS.mtFileClassification(inputFile, modelName, modelType, plotResults = False, gtFile = "")
	segs, classes = aS.flags2segs(flagsInd, mtStep)

	for i, s in enumerate(segs):
		if (classNames[int(classes[i])] == "Music") and (s[1] - s[0] >= minDuration):
			strOut = "{0:s}{1:.3f}-{2:.3f}.wav".format(dirOutput+os.sep, s[0], s[1])	
			wavfile.write( strOut, Fs, x[int(Fs*s[0]):int(Fs*s[1])])

def main(argv):	
	getMusicSegmentsFromFile(argv[1])	
	classifyFolderWrapper(argv[1][0:-4] + "_musicSegments", "svm", "data/svmMusicGenre6", True)		
	
	return 0
	
if __name__ == '__main__':
	main(sys.argv)
