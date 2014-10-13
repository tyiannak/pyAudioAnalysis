import sys, os, audioop, numpy, glob,  scipy, subprocess, wave, cPickle, threading, shutil, ntpath
import matplotlib.pyplot as plt
import audioFeatureExtraction as aF	
import audioTrainTest as aT
import audioSegmentation as aS
import audioVisualization as aV
import audioBasicIO
import utilities as uT
import scipy.io.wavfile as wavfile


def main(argv):
	if argv[1] == "-dirMp3toWAV":				# convert mp3 to wav (batch)
		if len(argv)==5:			
			path = argv[2]
			if argv[3] not in ["8000", "16000", "32000", "44100"]:
				print "Error. Unsupported sampling rate (must be: 8000, 16000, 32000 or 44100)."; return
			if argv[4] not in ["1","2"]:
				print "Error. Number of output channels must be 1 or 2"; return
			if not os.path.isdir(path):
				raise Exception("Input path not found!")
			useMp3TagsAsNames = True
			audioBasicIO.convertDirMP3ToWav(path, int(argv[3]), int(argv[4]), useMp3TagsAsNames)
		else:
			print "Error.\nSyntax: " + argv[0] + " -dirMp3toWAV <dirName> <sampling Freq> <numOfChannels>"

	elif argv[1] == "-featureExtractionFile":		# short-term and mid-term feature extraction to files (csv and numpy)
		if len(argv)==7:
			wavFileName = argv[2]
			if not os.path.isfile(wavFileName):
				raise Exception("Input audio file not found!")
			if not (uT.isNum(argv[3]) and uT.isNum(argv[4]) and uT.isNum(argv[5]) and uT.isNum(argv[6])):
				raise Exception("Mid-term and short-term window sizes and steps must be numbers!")
			mtWin = float(argv[3])
			mtStep = float(argv[4])
			stWin = float(argv[5])
			stStep = float(argv[6])
			outFile = wavFileName
			aF.mtFeatureExtractionToFile(wavFileName, mtWin, mtStep, stWin, stStep, outFile, True, True, True)
		else:
			print "Error.\nSyntax: " + argv[0] + " -featureExtractionFile <wavFileName> <mtWin> <mtStep> <stWin> <stStep>"

	elif argv[1] == '-featureExtractionDir':	# same as -featureExtractionFile, in a batch mode (i.e. for each WAV file in the provided path)
		if len(argv)==7:
			path = argv[2]
			if not os.path.isdir(path):
				raise Exception("Input path not found!")
			if not (uT.isNum(argv[3]) and uT.isNum(argv[4]) and uT.isNum(argv[5]) and uT.isNum(argv[6])):
				raise Exception("Mid-term and short-term window sizes and steps must be numbers!")
			mtWin = float(argv[3])
			mtStep = float(argv[4])
			stWin = float(argv[5])
			stStep = float(argv[6])
			aF.mtFeatureExtractionToFileDir(path, mtWin, mtStep, stWin, stStep, True, True, True)
		else:
			print "Error.\nSyntax: " + argv[0] + " -featureExtractionDir <path> <mtWin> <mtStep> <stWin> <stStep>"

	elif argv[1] == '-featureVisualizationDir':	# TODO dirsWavFeatureExtraction + dimensionality reduction (ffmpeg????)
		if len(argv)==3:
			aV.visualizeFeaturesFolder(argv[2], "pca", "artist")

	elif argv[1] == '-fileSpectrogram':		# show spectogram of a sound stored in a file
			if len(argv)==3:
				wavFileName = argv[2]		
				if not os.path.isfile(wavFileName):
					raise Exception("Input audio file not found!")
				[Fs, x] = audioBasicIO.readAudioFile(wavFileName)
				x = audioBasicIO.stereo2mono(x)
				specgram, TimeAxis, FreqAxis = aF.stSpectogram(x, Fs, round(Fs*0.020), round(Fs*0.020), True)
			else:
				print "Error.\nSyntax: " + argv[0] + " -fileSpectrogram <fileName>"

	elif argv[1] == '-fileChromagram':		# show spectogram of a sound stored in a file
			if len(argv)==3:
				wavFileName = argv[2]		
				if not os.path.isfile(wavFileName):
					raise Exception("Input audio file not found!")
				[Fs, x] = audioBasicIO.readAudioFile(wavFileName)
				x = audioBasicIO.stereo2mono(x)
				specgram, TimeAxis, FreqAxis = aF.stChromagram(x, Fs, round(Fs*0.040), round(Fs*0.040), True)
			else:
				print "Error.\nSyntax: " + argv[0] + " -fileSpectrogram <fileName>"


	elif argv[1] == '-speakerDiarization':		# speaker diarization (from file): TODO
			inputFile = argv[2]
			[Fs, x] = audioBasicIO.readAudioFile(inputFile)
			#speechLimits = aS.speechSegmentation(x, Fs, 2.0, 0.10, True)
			aS.speakerDiarization(x, Fs, 2.0, 0.1, int(argv[3]));
			#print speechLimits

	elif argv[1] == "-trainClassifier": 		# Segment classifier training (OK)
			if len(argv)>5: 
				method = argv[2]
				listOfDirs = argv[3:len(argv)-1]
				modelName = argv[-1]			
				aT.featureAndTrain(listOfDirs, 1, 1, aT.shortTermWindow, aT.shortTermStep, method.lower(), modelName)
			else:
				print "Error.\nSyntax: " + argv[0] + " -trainClassifier <method(svm or knn)> <directory 1> <directory 2> ... <directory N> <modelName>"

	elif argv[1] == "-classifyFile":		# Single File Classification (OK)
			if len(argv)==5: 
				modelType = argv[2]
				modelName = argv[3]
				inputFile = argv[4]

				if modelType not in ["svm", "knn"]:
					raise Exception("ModelType has to be either svm or knn!")
				if not os.path.isfile(modelName):
					raise Exception("Input modelName not found!")
				if not os.path.isfile(inputFile):
					raise Exception("Input audio file not found!")

				[Result, P, classNames] = aT.fileClassification(inputFile, modelName, modelType)
				print "{0:s}\t{1:s}".format("Class","Probability")
				for i,c in enumerate(classNames):
					print "{0:s}\t{1:.2f}".format(c,P[i])
				print "Winner class: " + classNames[int(Result)]
			else:
				print "Error.\nSyntax: " + argv[0] + " -classifyFile <method(svm or knn)> <modelName> <fileName>"

	elif argv[1] == "-classifyFolder": 			# Directory classification (Ok)
			if len(argv)==6 or len(argv)==5: 
				modelType = argv[2]
				modelName = argv[3]
				inputFolder = argv[4]
				if len(argv)==6:
					outputMode = argv[5]
				else:
					outputMode = "2"

				if modelType not in ["svm", "knn"]:
					raise Exception("ModelType has to be either svm or knn!")
				if outputMode not in ["0","1","2"]:
					raise Exception("outputMode has to be 0, 1 or 2")
				if not os.path.isfile(modelName):
					raise Exception("Input modelName not found!")
				if not os.path.isdir(inputFolder):
					raise Exception("Input folder not found!")

				types = ('*.wav',)
				wavFilesList = []
				for files in types:
					wavFilesList.extend(glob.glob(os.path.join(inputFolder, files)))

				wavFilesList = sorted(wavFilesList)	
				Results = []
				for wavFile in wavFilesList:	
					[Result, P, classNames] = aT.fileClassification(wavFile, modelName, modelType)	
					Result = int(Result)
					Results.append(Result)
					if outputMode=="1":
						print "{0:s}\t{1:s}".format(wavFile,classNames[Result])
				Results = numpy.array(Results)
				[Histogram, _] = numpy.histogram(Results, bins=numpy.arange(len(classNames)+1))
				if outputMode!="1":
					for h in Histogram:
						print "{0:20d}".format(h),
				if outputMode=="1":
					for i,h in enumerate(Histogram):
						print "{0:20s}\t\t{1:d}".format(classNames[i], h)
			else:
				print "Error.\nSyntax: " + argv[0] + " -classifyFolder <method(svm or knn)> <modelName> <folderName> <outputMode(0 or 1)"

	elif argv[1] == '-segmentClassifyFile':		# Segmentation-classification (OK)
		if (len(argv)==5):
			modelType = argv[2]
			modelName = argv[3]
			inputWavFile = argv[4]

			if modelType not in ["svm", "knn"]:
				raise Exception("ModelType has to be either svm or knn!")
			if not os.path.isfile(modelName):
				raise Exception("Input modelName not found!")
			if not os.path.isfile(inputWavFile):
				raise Exception("Input audio file not found!")

			[segs, classes] = aS.mtFileClassification(inputWavFile, modelName, modelType, True)
		else:
			print "Error.\nSyntax: " + argv[0] + " -segmentClassifyFile <method(svm or knn)> <modelName> <fileName>"

	elif argv[1] == '-thumbnail':			# music thumbnailing (OK)
			if len(argv)==4:	
				inputFile = argv[2]

				if not os.path.isfile(inputFile):
					raise Exception("Input audio file not found!")

				[Fs, x] = audioBasicIO.readAudioFile(inputFile)						# read file
				if Fs == -1:	# could not read file
					return
				try:
					thumbnailSize = float(argv[3])
				except ValueError:
					print "Thumbnail size must be a float (in seconds)"
					return 
				[A1, A2, B1, B2] = aS.musicThumbnailing(x, Fs, 1.0, 0.5, thumbnailSize)	# find thumbnail endpoints
				# write thumbnails to WAV files:
				thumbnailFileName1 = inputFile.replace(".wav","_thumb1.wav")
				thumbnailFileName2 = inputFile.replace(".wav","_thumb2.wav")
				wavfile.write(thumbnailFileName1, Fs, x[int(Fs*A1):int(Fs*A2)])
				wavfile.write(thumbnailFileName2, Fs, x[int(Fs*B1):int(Fs*B2)])
				print "1st thumbnail (stored in file {0:s}): {1:4.1f}sec -- {2:4.1f}sec".format(thumbnailFileName1, A1, A2)
				print "2nd thumbnail (stored in file {0:s}): {1:4.1f}sec -- {2:4.1f}sec".format(thumbnailFileName2, B1, B2)
			else: 
				print "Error.\nSyntax: " + argv[0] + " -thumbnail <filename> <thumbnailsize(seconds)>"

if __name__ == '__main__':
	main(sys.argv)
