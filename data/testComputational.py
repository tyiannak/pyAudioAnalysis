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
perTime1 =  duration / (t2-t1); print perTime1

t1 = time.clock()
aT.fileClassification("diarizationExample.wav", "svmSM","svm")
t2 = time.clock()
perTime1 =  duration / (t2-t1); print perTime1

t1 = time.clock()
[flagsInd, classesAll, acc] = aS.mtFileClassification("diarizationExample.wav", "svmSM", "svm", False, '')
t2 = time.clock()
perTime1 =  duration / (t2-t1); print perTime1

t1 = time.clock()
aS.hmmSegmentation('diarizationExample.wav', 'hmmRadioSM', False, '')             
t2 = time.clock()
perTime1 =  duration / (t2-t1); print perTime1

t1 = time.clock()
[Fs, x] = audioBasicIO.readAudioFile("diarizationExample.wav");
segments = aS.silenceRemoval(x, Fs, 0.050, 0.050, smoothWindow = 1.0, Weight = 0.3, plot = False)
t2 = time.clock()
perTime1 =  duration / (t2-t1); print perTime1


#plt.subplot(2,1,1); plt.plot(F[0,:]); plt.xlabel('Frame no'); plt.ylabel('ZCR'); 
#plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel('Energy'); plt.show()
