import audioTrainTest as aT
import audioSegmentation as aS

inputFile = "tempClock.wav"

modelType = "svm"
modelName = "data/svmMovies8classes"

[Fs, x] = audioBasicIO.readAudioFile(inputFile)

if modelType=='svm':
	[Classifier, MEAN, STD, classNames, mtWin, mtStep, stWin, stStep, computeBEAT] = aT.loadSVModel(modelName)
elif modelType=='knn':
	[Classifier, MEAN, STD, classNames, mtWin, mtStep, stWin, stStep, computeBEAT] = aT.loadKNNModel(modelName)

flagsInd, classNames, acc = aS.mtFileClassification(inputFile, modelName, modelType, plotResults = False, gtFile = "")
segs, classes = aS.flags2segs(flagsInd, mtStep)
for i in range(len(segs)):
	print segs[i,0], segs[i,1], classNames[int(classes[i])]
