import audioFeatureExtraction as a
import numpy
import audioBasicIO
import matplotlib.pyplot as plt
import utilities

def run(F):
	toWatch = [0,1,3,5,6,7,8]
	HistAll = numpy.zeros((50,));
	for i in toWatch:	
		DifThres = 2.0*(numpy.abs(F[i,0:-1] - F[i,1::])).mean()
		[pos1, _] = utilities.peakdet(F[i,:], DifThres)
		plt.clf()
		plt.subplot(3,1,1);plt.plot(F[i,:])
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
		plt.subplot(3,1,2); plt.plot(HistCenters, HistTimes)
		plt.text(HistCenters[HistCenters.shape[0]/2],0, str(i))
		HistAll += HistTimes
		plt.subplot(3,1,3);plt.plot(HistCenters, HistAll)
		plt.show(block=False)
		plt.draw()
	plt.clf()
	plt.plot(HistCenters, HistAll);
	plt.show(block=True)
def plotAll(F):
	for i in range(F.shape[0]):
		DifThres = 2.0*(numpy.abs(F[i,0:-1] - F[i,1::])).mean()
		[pos1, _] = utilities.peakdet(F[i,:], DifThres)
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

[Fs, x] = audioBasicIO.readAudioFile("New Order - True Faith [OFFICIAL MUSIC VIDEO].wav");
#[Fs, x] = audioBasicIO.readAudioFile("Trentemoller.wav");
#x = x[Fs*10:Fs*70]
F = a.stFeatureExtraction(x, Fs, 0.050*Fs, 0.050*Fs);
run(F)


