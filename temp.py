import audioFeatureExtraction as a
import numpy
import audioBasicIO
import matplotlib.pyplot as plt
import utilities

def beatExtraction(stFeatures, winSize):
	toWatch = [0,1,3,5,6,7,8]
	maxBeatTime = int(round(1.0 / winSize));
	HistAll = numpy.zeros((maxBeatTime,));
	for i in toWatch:	
		DifThres = 2.0*(numpy.abs(F[i,0:-1] - F[i,1::])).mean()
		[pos1, _] = utilities.peakdet(F[i,:], DifThres)

		posDifs = []
		for j in range(len(pos1)-1):
			posDifs.append(pos1[j+1]-pos1[j])
		[HistTimes, HistEdges] = numpy.histogram(posDifs, numpy.arange(0.5, maxBeatTime + 1.5))
		HistCenters = (HistEdges[0:-1] + HistEdges[1::]) / 2.0
		HistTimes = HistTimes.astype(float) / F.shape[1]
		HistAll += HistTimes
		plt.clf()
		plt.subplot(3,1,1);plt.plot(F[i,:])
		for k in pos1:
			plt.plot(k, F[i, k], '*')
		plt.subplot(3,1,2); plt.plot(HistCenters, HistTimes)
		plt.text(HistCenters[HistCenters.shape[0]/2],0, str(i))
		plt.subplot(3,1,3);plt.plot(HistCenters, HistAll)
		#plt.show(block=False)
		plt.show()
		plt.draw()
	plt.clf()
	plt.plot(60/(HistCenters * winSize), HistAll);
	plt.show(block=True)

def plotAll(F):
	for i in range(F.shape[0]):
		DifThres = 2.0*(numpy.abs(F[i,0:-1] - F[i,1::])).mean()
		#[pos1, _] = utilities.peakdet(F[i,:], DifThres)
		[pos1, _] = utilities.peakdet(stFeatures[i,:], DifThres, step = 10)
		plt.subplot(2,1,1);plt.plot(F[i,:])
		for k in pos1:
			plt.plot(k, F[i, k], '*')

		posDifs = []
		for j in range(len(pos1)-1):
			posDifs.append(pos1[j+1]-pos1[j])
		[HistTimes, HistEdges] = numpy.histogram(posDifs, numpy.arange(0.5,51))

		HistCenters = (HistEdges[0:-1] + HistEdges[1::]) / 2.0
		print HistCenters
		HistTimes = HistTimes.astype(float) / F.shape[1]
		print HistCenters.shape, HistTimes.shape
		plt.subplot(2,1,2); plt.plot(HistCenters, HistTimes)
		plt.text(HistCenters[HistCenters.shape[0]/2],0, str(i))
		plt.show()

[Fs, x] = audioBasicIO.readAudioFile("170 BPM - Simple Straight Beat - Drum Track.wav");
#[Fs, x] = audioBasicIO.readAudioFile("Trentemoller.wav");
x = x[Fs*50:Fs*100]
F = a.stFeatureExtraction(x, Fs, 0.050*Fs, 0.050*Fs);
beatExtraction(F, 0.050)



