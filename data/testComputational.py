from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioSegmentation as aS
import matplotlib.pyplot as plt
import time

[Fs, x] = audioBasicIO.readAudioFile("diarizationExample.wav");
duration = x.shape[0] / float(Fs)
t1 = time.clock()
F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.050*Fs);
t2 = time.clock()
perTime1 =  duration / (t2-t1); print "short-term feature extraction: {0:.1f} x realtime".format(perTime1)

t1 = time.clock()
aT.fileClassification("diarizationExample.wav", "svmSM","svm")
t2 = time.clock()
perTime1 =  duration / (t2-t1); print "Mid-term feature extraction + classification: {0:.1f} x realtime".format(perTime1)

t1 = time.clock()
[flagsInd, classesAll, acc] = aS.mtFileClassification("diarizationExample.wav", "svmSM", "svm", False, '')
t2 = time.clock()
perTime1 =  duration / (t2-t1); print "Fix-sized classification - segmentation: {0:.1f} x realtime".format(perTime1)

t1 = time.clock()
aS.hmmSegmentation('diarizationExample.wav', 'hmmRadioSM', False, '')             
t2 = time.clock()
perTime1 =  duration / (t2-t1); print "HMM-based classification - segmentation: {0:.1f} x realtime".format(perTime1)

t1 = time.clock()
[Fs, x] = audioBasicIO.readAudioFile("diarizationExample.wav");
segments = aS.silenceRemoval(x, Fs, 0.050, 0.050, smoothWindow = 1.0, Weight = 0.3, plot = False)
t2 = time.clock()
perTime1 =  duration / (t2-t1); print "Silence removal: {0:.1f} x realtime".format(perTime1)

"""
t1 = time.clock()
[Fs1, x1] = audioBasicIO.readAudioFile("../MusicData/Amy Winehouse --- Me & Mr Jones.wav")
[A1, A2, B1, B2, Smatrix] = aS.musicThumbnailing(x1, Fs1, 1.0, 1.0, 15.0)	# find thumbnail endpoints			
duration1 = x1.shape[0] / float(Fs1)
t2 = time.clock()
perTime1 =  duration1 / (t2-t1); print "Thumbnail: {0:.1f} x realtime".format(perTime1)
"""


#plt.subplot(2,1,1); plt.plot(F[0,:]); plt.xlabel('Frame no'); plt.ylabel('ZCR'); 
#plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel('Energy'); plt.show()
