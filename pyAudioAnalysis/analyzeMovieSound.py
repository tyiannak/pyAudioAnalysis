import os, sys, shutil, glob, numpy, csv, cPickle
import scipy.io.wavfile as wavfile
import audioBasicIO
import audioTrainTest as aT
import audioSegmentation as aS
import matplotlib.pyplot as plt
import scipy.spatial.distance
minDuration = 7;

def classifyFolderWrapper(inputFolder, modelType, modelName, outputMode=False):
	if not os.path.isfile(modelName):
		raise Exception("Input modelName not found!")

	if modelType=='svm':
		[Classifier, MEAN, STD, classNames, mtWin, mtStep, stWin, stStep, compute_beat] = aT.load_model(modelName)
	elif modelType=='knn':
		[Classifier, MEAN, STD, classNames, mtWin, mtStep, stWin, stStep, compute_beat] = aT.load_model_knn(modelName)

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
	if outputMode:	
		for i,h in enumerate(Histogram):
			print "{0:20s}\t\t{1:d}".format(classNames[i], h)
	PsAll = PsAll / numpy.sum(PsAll)


	if outputMode:	
		fig = plt.figure()
		ax = fig.add_subplot(111)
		plt.title("Classes percentage " + inputFolder.replace('Segments',''))
		ax.axis((0, len(classNames)+1, 0, 1))
		ax.set_xticks(numpy.array(range(len(classNames)+1)))
		ax.set_xticklabels([" "] + classNames)
		ax.bar(numpy.array(range(len(classNames)))+0.5, PsAll)
		plt.show()
	return classNames, PsAll

def getMusicSegmentsFromFile(inputFile):	
	modelType = "svm"
	modelName = "data/svmMovies8classes"
	
	dirOutput = inputFile[0:-4] + "_musicSegments"
	
	if os.path.exists(dirOutput) and dirOutput!=".":
		shutil.rmtree(dirOutput)	
	os.makedirs(dirOutput)	
	
	[Fs, x] = audioBasicIO.readAudioFile(inputFile)	

	if modelType=='svm':
		[Classifier, MEAN, STD, classNames, mtWin, mtStep, stWin, stStep, compute_beat] = aT.load_model(modelName)
	elif modelType=='knn':
		[Classifier, MEAN, STD, classNames, mtWin, mtStep, stWin, stStep, compute_beat] = aT.load_model_knn(modelName)

	flagsInd, classNames, acc, CM = aS.mtFileClassification(inputFile, modelName, modelType, plotResults = False, gtFile = "")
	segs, classes = aS.flags2segs(flagsInd, mtStep)

	for i, s in enumerate(segs):
		if (classNames[int(classes[i])] == "Music") and (s[1] - s[0] >= minDuration):
			strOut = "{0:s}{1:.3f}-{2:.3f}.wav".format(dirOutput+os.sep, s[0], s[1])	
			wavfile.write( strOut, Fs, x[int(Fs*s[0]):int(Fs*s[1])])

def analyzeDir(dirPath):
	for i,f in enumerate(glob.glob(dirPath + os.sep + '*.wav')):				# for each WAV file					
		getMusicSegmentsFromFile(f)	
		[c, P]= classifyFolderWrapper(f[0:-4] + "_musicSegments", "svm", "data/svmMusicGenre8", False)
		if i==0:
			print "".ljust(100)+"\t",
			for C in c:
				print C.ljust(12)+"\t",
			print
		print f.ljust(100)+"\t",
		for p in P:
				print "{0:.2f}".format(p).ljust(12)+"\t",
		print
		
def main(argv):	
	
	if argv[1]=="--file":
		getMusicSegmentsFromFile(argv[2])	
		classifyFolderWrapper(argv[2][0:-4] + "_musicSegments", "svm", "data/svmMusicGenre8", True)		
		
	elif argv[1]=="--dir":	
		analyzeDir(argv[2])	
		
	elif argv[1]=="--sim":
		csvFile = argv[2]
		f = []
		fileNames = []
		with open(csvFile, 'rb') as csvfile:
			spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
			for j,row in enumerate(spamreader):
				if j>0:
					ftemp = []
					for i in range(1,9):
						ftemp.append(float(row[i]))
					f.append(ftemp)
					R = row[0]
					II = R.find(".wav");
					fileNames.append(row[0][0:II])
			f = numpy.array(f)

			Sim = numpy.zeros((f.shape[0], f.shape[0]))
			for i in range(f.shape[0]):
				for j in range(f.shape[0]):	
					Sim[i,j] = scipy.spatial.distance.cdist(numpy.reshape(f[i,:], (f.shape[1],1)).T, numpy.reshape(f[j,:], (f.shape[1],1)).T, 'cosine')
								
			Sim1 = numpy.reshape(Sim, (Sim.shape[0]*Sim.shape[1], 1))
			plt.hist(Sim1)
			plt.show()

			fo = open(csvFile + "_simMatrix", "wb")
			cPickle.dump(fileNames,  fo, protocol = cPickle.HIGHEST_PROTOCOL)
			cPickle.dump(f, fo, protocol = cPickle.HIGHEST_PROTOCOL)			
			cPickle.dump(Sim, fo, protocol = cPickle.HIGHEST_PROTOCOL)
			fo.close()

	elif argv[1]=="--loadsim":
		try:
			fo = open(argv[2], "rb")
		except IOError:
				print "didn't find file"
				return
		try:			
			fileNames 	= cPickle.load(fo)
			f 			= cPickle.load(fo)
			Sim 		= cPickle.load(fo)
		except:
			fo.close()
		fo.close()	
		print fileNames
		Sim1 = numpy.reshape(Sim, (Sim.shape[0]*Sim.shape[1], 1))
		plt.hist(Sim1)
		plt.show()

	elif argv[1]=="--audio-event-dir":		
		files = "*.wav"
		inputFolder = argv[2]
		if os.path.isdir(inputFolder):
			strFilePattern = os.path.join(inputFolder, files)
		else:
			strFilePattern = inputFolder + files

		wavFilesList = []
		wavFilesList.extend(glob.glob(strFilePattern))
		wavFilesList = sorted(wavFilesList)		
		for i,w in enumerate(wavFilesList):			
			[flagsInd, classesAll, acc, CM] = aS.mtFileClassification(w, "data/svmMovies8classes", "svm", False, '')
			histTemp = numpy.zeros( (len(classesAll), ) )
			for f in flagsInd:
				histTemp[int(f)] += 1.0
			histTemp /= histTemp.sum()
			
			if i==0:
				print "".ljust(100)+"\t",
				for C in classesAll:
					print C.ljust(12)+"\t",
				print
			print w.ljust(100)+"\t",
			for h in histTemp:				
				print "{0:.2f}".format(h).ljust(12)+"\t",
			print

			
	return 0
	
if __name__ == '__main__':
	main(sys.argv)
