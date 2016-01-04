# pyAudioAnalysis: A Python Audio Analysis Library

pyAudioAnalysis is a Python library covering a wide range of audio analysis tasks, including: feature extraction, classification, segmentation and visualization. 
 The user can perform the following tasks:
 * Extract a wide range of audio features and representations (e.g. spectrogram, chromagram)
 * Train, parameter tune and evaluate segment-based classifiers
 * Classify unknown samples
 * Detect audio events and exclude silence periods from long recordings
 * Perform supervised segmentation (i.e. apply a clasification model on fix-sized segments)
 * Perform unsupervised segmentation (e.g. speaker diarization)
 * Extract audio thumbnails
 * Train and use audio regression models (example application: emotion recognition)
 * Apply dimensionality reduction techniques to visualize audio data and content similarities


*[(NEW: PLOS One Paper regarding pyAudioAnalysis)] (http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0144610)*

*[(follow this link for the complete documentation)] (https://github.com/tyiannak/pyAudioAnalysis/wiki)*

*pyAudioAnalysis can serve as an introduction to Audio Analysis in Python, for Matlab-related audio analysis material check  [this book](http://www.amazon.com/Introduction-Audio-Analysis-MATLAB%C2%AE-Approach/dp/0080993885).*


pyAudioAnalysis provides easy-to-call wrappers to execute audio analysis tasks. For example, to train a classifier segments, given a set of WAV files stored in folders, each folder representing a different class,
 the following code needs to be executed:

```
from pyAudioAnalysis import audioTrainTest as aT
aT.featureAndTrain(["classifierData/music","classifierData/speech"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmSMtemp", False)
```

Then, the resulting classification model can be used to classify an unknown audio WAV file:
```
aT.fileClassification("data/doremi.wav", "svmSMtemp","svm")
Result:
(0.0, array([ 0.90156761,  0.09843239]), ['music', 'speech'])
```

In addition, command-line support is provided for all functionalities. E.g. the following command needs to be executed to extract the spectrogram of an audio signal stored in a WAV file:
```
python audioAnalysis.py fileSpectrogram -i data/doremi.wav
```

*[Installing instructions and a complete documentation is provided in the wiki] (https://github.com/tyiannak/pyAudioAnalysis/wiki)*

*Author: [Theodoros Giannakopoulos]*


