import sys, os, audioop, numpy, glob,  scipy, subprocess, wave, mlpy, cPickle, threading, shutil
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from scipy.fftpack import rfft
import audioFeatureExtraction as aF	
import audioTrainTest as aT
import audioSegmentation as aS
from scipy.fftpack import fft

def main(argv):
	if argv[1] == '-fileSpectrogram':		# show spectogram of a sound stored in a file
			if len(argv)==3:
				wavFileName = argv[2]		
				if not os.path.isfile(wavFileName):
					raise Exception("Input audio file not found!")
				[Fs, x] = aF.readAudioFile(wavFileName)
				x = aF.stereo2mono(x)
				specgram, TimeAxis, FreqAxis = aF.stSpectogram(x, Fs, round(Fs*0.020), round(Fs*0.020), True)
			else:
				print "Error.\nSyntax: " + argv[0] + " -fileSpectrogram <fileName>"

	elif argv[1] == '-speakerDiarization':		# speaker diarization (from file): TODO
			inputFile = argv[2]
			[Fs, x] = aF.readAudioFile(inputFile)
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


				[Fs, x] = aF.readAudioFile(inputFile)						# read file
				if Fs == -1:	# could not read file
					return
				try:
					thumbnailSize = float(argv[3])
				except ValueError:
					print "Thumbnail size must be a float (in seconds)"
					return 
				[A1, A2, B1, B2] = aS.musicThumbnailing(x, Fs, 1.0, 0.5, thumbnailSize)		# find thumbnail endpoints
				# write thumbnails to WAV files:
				thumbnailFileName1 = inputFile.replace(".wav","_thumb1.wav")
				thumbnailFileName2 = inputFile.replace(".wav","_thumb2.wav")
				wavfile.write(thumbnailFileName1, Fs, x[round(Fs*A1):round(Fs*A2)])
				wavfile.write(thumbnailFileName2, Fs, x[round(Fs*B1):round(Fs*B2)])
				print "1st thumbnail (stored in file {0:s}): {1:4.1f}sec -- {2:4.1f}sec".format(thumbnailFileName1, A1, A2)
				print "2nd thumbnail (stored in file {0:s}): {1:4.1f}sec -- {2:4.1f}sec".format(thumbnailFileName2, B1, B2)
			else: 
				print "Error.\nSyntax: " + argv[0] + " -thumbnail <filename> <thumbnailsize(seconds)>"

	
if __name__ == '__main__':
	main(sys.argv)
