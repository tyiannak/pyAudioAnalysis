# pyAudioAnalysis: A Python Audio Analysis Library

pyAudioAnalysis is a Python library covering a wide range of audio analysis tasks, including: feature extraction, classification, segmentation and visualization. 

pyAudioAnalysis provides easy-to-call wrappers to execute audio analysis tasks. For example, to train a classifier segments, given a set of WAV files stored in folders, each folder representing a different class,
 the following code needs to be executed:

```
from pyAudioAnalysis import audioTrainTest as aT
aT.featureAndTrain(["classifierData/music","classifierData/speech"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmSMtemp", False)
```

Then, the resulting classification model can be used to classify an unknown audio WAV file:
```
XXXXX
Result:
YYYYY
```

In addition, command-line support is provided for all functionalities. E.g. the following command needs to be executed to extract the spectrogram of an audio signal stored in a WAV file:
```
python audioAnalysis.py -fileSpectrogram data/doremi.wav
```

*[A complete Wiki documentation is provided here] (https://github.com/tyiannak/pyAudioAnalysis/wiki)*

*Author: [Theodoros Giannakopoulos]*


