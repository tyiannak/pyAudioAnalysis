from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS
import matplotlib.pyplot as plt
[Fs, x] = aIO.readAudioFile("data/recording1.wav");
segments = aS.silenceRemoval(x, Fs, 0.020, 0.020, smoothWindow = 1.0, Weight = 0.3, plot = True)
print segments



