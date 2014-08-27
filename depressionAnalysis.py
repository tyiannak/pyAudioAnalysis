import sys, os
import audioFeatureExtraction as aF	
import audioTrainTest as aT
import fnmatch
import matplotlib.pyplot as plt
import numpy, mlpy

def changeAR(dirName):
	for fname in os.listdir(dirName):
	    	if fnmatch.fnmatch(fname, '*_*_cut_audio*.wav'):
			command = "ffmpeg -ar 16000 -ac 1 " + fname
			print command
	

def getDepressionFeatures(dirName):
	fileTuples = []
	for fname in os.listdir(dirName):
	    	if fnmatch.fnmatch(fname, '*_*_*.wav'):			
		        # print fname
			a = fname.split('_');
			speakerID = a[0]
			fileID	  = a[1]
			strGT = dirName + speakerID + '_' + fileID + '_Depression.csv'

			f = open(strGT, 'r')
			rData = f.read();
			BDI = int(rData)
			f.close()

			fileTuples.append(( dirName + fname, int(speakerID), int(fileID), BDI) )

	fileTuples = sorted(fileTuples, key=lambda fileT: fileT[0])

	BIDs = [x[3] for x in fileTuples]

	fig = plt.figure()
	ax = fig.add_subplot(111)

	n, bins, patches = ax.hist(BIDs, 20)
	plt.show()

	Features = []
	Ys = []
	# feature extraction
	for i in range(len(fileTuples)):
		f = fileTuples[i]
		print "Processing file " + str(i) + " of " + str(len(fileTuples))
		[Fs, x] = aF.readAudioFile(f[0])
		[FF, _] = aF.mtFeatureExtraction(x, Fs, 10.0*Fs, 1.0*Fs, 0.050*Fs, 0.050*Fs)
		Features.append(numpy.transpose(FF))
		Ys.append(f[3])	
	return (Features, Ys)

def trainSVMregression(Features, Ys, C):
	[X, Y] = aT.listOfFeatures2MatrixRegression(Features, Ys)
	#svm = mlpy.LibSvm(svm_type='nu_svr', kernel_type='linear', nu=Nu, eps=0.0000001, gamma = Gamma)
	svm = mlpy.LibSvm(svm_type='c_svc', kernel_type='linear', eps=0.0000001, C = C, probability=True)
	svm.learn(X, Y)
	Results = svm.pred(X)
	#print Results
	#print numpy.mean(abs(Results-Y))

	return svm 

def estimateDepression(svm, Features, MEAN, STD):	
	featuresNorm = []	
	ft = Features.copy()
	for nSamples in range(Features.shape[0]):
		ft[nSamples,:] = (ft[nSamples,:] - MEAN) / STD
	Results = svm.pred(ft)
	# TODO: something else instead of simple averaging???
	return numpy.mean(Results)

def evaluateDepressionScript(Features, BDI, partTrain, C, nExp):
	Etrains = []
	Etests = []
	Acs = []
	AcBins = []
	for i in range(nExp):
		[Etrain, Etest, Ac, AcBin] = evaluateDepression(Features, BDI, partTrain, C)
		Etrains.append(Etrain)
		Etests.append(Etest)
		Acs.append(Ac)
		AcBins.append(AcBin)
	Etrains = numpy.array(Etrains)
	Etests = numpy.array(Etests)	
	Acs = numpy.array(Acs)
	AcBins = numpy.array(AcBins)

	print C, numpy.median(Etrains), numpy.median(Etests), numpy.median(Acs), numpy.median(AcBins)

def evaluateDepressionScriptC(Features, BDI, partTrain):
	Cs = [0.001, 0.005, 0.0075, 0.1, 0.25, 0.5, 1.0];
	#Cs = [0.05, 0.25, 1.5 ];
	for i, c in enumerate(Cs):
		evaluateDepressionScript(Features,BDI, partTrain, c, 200)

def findDepressionClass(BDI_value):	
	DLimits = [0, 14, 20, 29, 65];
	for i, d in enumerate(DLimits):
		if BDI_value<d:
			break;
	return i-1

def evaluateDepression(Features, BDI, partTrain, C):
	nFiles = len(Features)
	
	randperm = numpy.random.permutation(range(nFiles))

	nTrain = int(round(partTrain * nFiles))
	featuresTrain = [Features[randperm[i]] for i in range(nTrain)]
	featuresTest = [Features[randperm[i+nTrain]] for i in range(nFiles - nTrain)]
	BDIsTrain = [BDI[randperm[i]] for i in range(nTrain)]
	BDIsTest  = [BDI[randperm[i+nTrain]] for i in range(nFiles - nTrain)]

	(featuresNorm, MEAN, STD) = aT.normalizeFeatures(featuresTrain)

	svm = trainSVMregression(featuresNorm, BDIsTrain, C)
	ErrorTrain = []
	ErrorTest = []
	#print "TEST"
	CM = numpy.zeros((4,4))

	for i in range(len(featuresTest)):
		R = estimateDepression(svm, featuresTest[i], MEAN, STD)
		ErrorTest.append((R - BDIsTest[i])*(R - BDIsTest[i]));
		iTrue = findDepressionClass(BDIsTest[i])
		iTest = findDepressionClass(R)
		CM[iTrue,iTest] = CM[iTrue,iTest] + 1.0
		#print R, BDIsTest[i]

	#print "TRAIN"
	for i in range(len(featuresTrain)):
		R = estimateDepression(svm, featuresTrain[i], MEAN, STD)
		ErrorTrain.append((R - BDIsTrain[i])*(R - BDIsTrain[i]));
		# print R, BDIsTrain[i]

	return (numpy.sqrt(numpy.mean(ErrorTrain)),  numpy.sqrt(numpy.mean(ErrorTest)), numpy.sum(numpy.diagonal(CM)) / numpy.sum(CM), (numpy.sum(CM[0:2,0:2])+numpy.sum(CM[2:4,2:4])) / numpy.sum(CM))


#	featuresTest = Features[randperm[nTrain+1:-1]]
	
	

def main(argv):
	[Features, Ys] = getDepressionFeatures(dirName)
	#X, Y = aT.listOfFeatures2MatrixRegression(Features, Ys)
	#print X.shape, Y.shape

if __name__ == '__main__':
	main(sys.argv)
