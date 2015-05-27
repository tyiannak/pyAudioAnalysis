import scipy.io.wavfile as wavfile
import audioBasicIO
import audioTrainTest as aT
import audioSegmentation as aS

inputFile = "tempClock.wav"
modelType = "svm"
modelName = "data/svmMovies8classes"

minDuration = 10;

[Fs, x] = audioBasicIO.readAudioFile(inputFile)

if modelType=='svm':
	[Classifier, MEAN, STD, classNames, mtWin, mtStep, stWin, stStep, computeBEAT] = aT.loadSVModel(modelName)
elif modelType=='knn':
	[Classifier, MEAN, STD, classNames, mtWin, mtStep, stWin, stStep, computeBEAT] = aT.loadKNNModel(modelName)

flagsInd, classNames, acc = aS.mtFileClassification(inputFile, modelName, modelType, plotResults = False, gtFile = "")
segs, classes = aS.flags2segs(flagsInd, mtStep)

for i in range(len(segs)):
	print segs[i,0], segs[i,1], classNames[int(classes[i])]

for i, s in enumerate(segs):
	if (classNames[int(classes[i])] == "Music") and (s[1] - s[0] > minDuration):
		strOut = "{0:s}_{1:.3f}-{2:.3f}.wav".format(inputFile[0:-4], s[0], s[1])
		print strOut
		wavfile.write( strOut, Fs, x[int(Fs*s[0]):int(Fs*s[1])])
