<img src="icon.png" align="left" height="70"/>
A Python Library for Audio Analysis: 
Feature Extraction, Classification, Segmentation and Applications
*This doc contains general info. Follow [this link] (https://github.com/tyiannak/pyAudioAnalysis/wiki) for the complete documentation*

## News
 * August 2016: pyAudioAnalysis has been updated. mlpy is no longer used and all learning tasks (svm, kmeans, pcm, lda) are performed through the scikit-learn package. Also, dependencies have been simplified (see documentation for details)
 * January 2016: *[PLOS-One Paper regarding pyAudioAnalysis (please cite!)] (http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0144610)*

## General
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


## Installation
 * Install dependencies:
 ```
pip install numpy matplotlib scipy sklearn hmmlearn simplejson eyed3
```
 * Clone source: 
 ```
git clone https://github.com/tyiannak/pyAudioAnalysis.git
```

## An audio classification example
> More examples and detailed tutorials can be found [at the wiki] (https://github.com/tyiannak/pyAudioAnalysis/wiki)

pyAudioAnalysis provides easy-to-call wrappers to execute audio analysis tasks. Eg, this code trains an audio segment classifier, given a set of WAV files stored in folders (each folder representing a different class):

```
from pyAudioAnalysis import audioTrainTest as aT
aT.featureAndTrain(["classifierData/music","classifierData/speech"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmSMtemp", False)
```

Then, the trainned classifier can be used to classify an unknown audio WAV file:
```
aT.fileClassification("data/doremi.wav", "svmSMtemp","svm")
Result:
(0.0, array([ 0.90156761,  0.09843239]), ['music', 'speech'])
```

In addition, command-line support is provided for all functionalities. E.g. the following command extracts the spectrogram of an audio signal stored in a WAV file:
```
python audioAnalysis.py fileSpectrogram -i data/doremi.wav
```

*pyAudioAnalysis can serve as an introduction to Audio Analysis in Python, for Matlab-related audio analysis material check  [this book](http://www.amazon.com/Introduction-Audio-Analysis-MATLAB%C2%AE-Approach/dp/0080993885).*

*Author: [Theodoros Giannakopoulos]*


