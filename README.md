# pyAudioAnalysis: A Python Audio Analysis Library

## General
pyAudioAnalysis is a python library for basic audio analysis tasks, including: feature extraction, classification, segmentation and visualization. 

*Author: [Theodoros Giannakopoulos]*

## Download
Type the following in your terminal:  
```
git clone https://github.com/tyiannak/pyAudioAnalysis.git
```

## Dependencies
Below you can find a list of library dependencies, along with the Linux commands to install them. 
 * MLPY
```
wget http://sourceforge.net/projects/mlpy/files/mlpy%203.5.0/mlpy-3.5.0.tar.gz
tar xvf mlpy-3.5.0.tar.gz
cd mlpy-3.5.0
sudo python setup.py install
```
 * NUMPY
 ```
sudo apt-get install python-numpy
```
 * MATPLOTLIB
 ```
sudo apt-get install python-matplotlib
```
 * SCIPY 
```
sudo apt-get install python-scipy
```
 * GSL
```
sudo apt-get install libgsl0-dev
```
 * AlsaAudio
```
sudo apt-get install python-alsaaudio
```

## General Structure
The library code is organized in 4 Python files. In particular:
 * `audioAnalysis.py`: this file implements the command-line interface of the basic functionalities of the library, along with some recording functionalities.
 * `audioFeatureExtraction.py`: this is where all audio feature extraction is implemented. In total, 21 short-term features are computed, while a mid-term windowing technique is also implemented, in order to extract statistics of audio features. 
 * `audioTrainTest.py`: this file implements the audio classification prodecures. It contains functions that can be used to train a Support Vector Machine or k-Nearest-Neighbour classifier. Also, wrapper functions and scripts are provided for general training, evaluating and feature normalization issues. 
 * `audioSegmentation.py`: this file implements audio segmentation functionalities, e.g. fixed-sized segment classification and segmentation, speaker diarization, etc. 

In the `data/` folder, a couple of audio sample files are provided, along with some trained SVM and kNN models for particular classification tasks (e.g. Speech vs Music, Musical Genre Classification, etc).

## Basic Functionalities

### Record fix-sized audio segments
Function `recordAudioSegments(RecordPath, BLOCKSIZE)` from the `audioAnalysis.py` file.

Command-line use example: 
```
python audioAnalysis.py -recordSegments "rSpeech" 2.0
```

### Realtime fix-sized segments classification
Function `recordAnalyzeAudio(duration, outputWavFile, midTermBufferSizeSec, modelName, modelType)` from the `audioAnalysis.py` file. 

Command-line use example 
```
python audioAnalysis.py -recordAndClassifySegments 20 out.wav knnRecNoiseActivity knn
```

### Train Segment Classifier From Data
A segment classification functionality is provided in the library. Towards this end, the `audioTrainTest.py` file implements two types of classifiers, namelly the kNN and SVM methods. Below, we describe how to train a segment classifier from data (i.e. segments stored in WAV files, organized in directories that correspond to classes).


The function used to train a segment classifier model is `featureAndTrain()` from `audioTrainTest.py`. Example:
```
import audioTrainTest as aT
aT.featureAndTrain(["/home/tyiannak/Desktop/MusicGenre/Classical/","/home/tyiannak/Desktop/MusicGenre/Electronic/","/home/tyiannak/Desktop/MusicGenre/Jazz/"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmMusicGenre3")
aT.featureAndTrain(["/home/tyiannak/Desktop/MusicGenre/Classical/","/home/tyiannak/Desktop/MusicGenre/Electronic/","/home/tyiannak/Desktop/MusicGenre/Jazz/"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "knn", "knnMusicGenre3")
aT.featureAndTrain(["/home/tyiannak/Desktop/5Class/Silence/","/home/tyiannak/Desktop/5Class/SpeechMale/","/home/tyiannak/Desktop/5Class/SpeechFemale/","/home/tyiannak/Desktop/5Class/ObjectsOther/","/home/tyiannak/Desktop/5Class/Music/"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svm5Classes")
aT.featureAndTrain(["/home/tyiannak/Desktop/5Class/Silence/","/home/tyiannak/Desktop/5Class/SpeechMale/","/home/tyiannak/Desktop/5Class/SpeechFemale/","/home/tyiannak/Desktop/5Class/ObjectsOther/","/home/tyiannak/Desktop/5Class/Music/"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "knn", "knn5Classes")
```
Command-line use:
```
python audioAnalysis.py -trainClassifier <method(svm or knn)> <directory 1> <directory 2> ... <directory N> <modelName>`. 
```
Examples:
```
python audioAnalysis.py -trainClassifier svm /home/tyiannak/Desktop/SpeechMusic/music /home/tyiannak/Desktop/SpeechMusic/speech data/svmSM
```
```
python audioAnalysis.py -trainClassifier knn /home/tyiannak/Desktop/ /home/tyiannak/Desktop/SpeechMusic/speech data/knnSM
```
```
python audioAnalysis.py -trainClassifier knn /home/tyiannak/Desktop/MusicGenre/Classical/ /home/tyiannak/Desktop/MusicGenre/Electronic/ /home/tyiannak/Desktop/MusicGenre/Jazz/  data/knnMusicGenre3
```
```
python audioAnalysis.py -trainClassifier svm /home/tyiannak/Desktop/MusicGenre/Classical/ /home/tyiannak/Desktop/MusicGenre/Electronic/ /home/tyiannak/Desktop/MusicGenre/Jazz/  data/svmMusicGenre3
```

### Single File Classification

Function `fileClassification(inputFile, modelName, modelType)` from `audioTrainTest.py` file can be used to classify a single wav file based on an already trained segment classifier. 

Example:
```
import audioTrainTest as aT
aT.fileClassification("TrueFaith.wav", "data/svmMusicGenre3","svm")
```

Command-line use:
```
python audioAnalysis.py -classifyFile <method(svm or knn)> <modelName> <fileName>
```

Examples:
```
python audioAnalysis.py -classifyFile knn data/knnSM data/TrueFaith.wav
python audioAnalysis.py -classifyFile knn data/knnMusicGenre3 data/TrueFaith.wav
python audioAnalysis.py -classifyFile svm data/svmMusicGenre3 data/TrueFaith.wav
```

### Folder Classification
Classifies each WAV file found in the given folder and generates stdout resutls:
Command-line use examples:
```
python audioAnalysis.py -classifyFolder svm data/svmSM RecSegments/Speech/ 0 (only generates freq counts for each audio class)
python audioAnalysis.py -classifyFolder svm data/svmSM RecSegments/Speech/ 1 (also outputs the result of each singe WAV file)
```

### File Segmentation & Classification
Function		`mtFileClassification` from `audioSegmentation.py`.

Example:
```
import audioSegmentation as aS
[segs, classes] = aS.mtFileClassification("data/speech_music_sample.wav", "data/svmSM", "svm", True)
```
Command-line use:
```
python audioAnalysis.py -segmentClassifyFile <method(svm or knn)> <modelName> <fileName>
```
Example:
```
python audioAnalysis.py -segmentClassifyFile svm data/svmSM data/speech_music_sample.wav 
```

### Audio thumbnailing

[Audio thumbnailing] is an important application of music information retrieval that focuses on detecting instances of the most representative part of a music recording. In `pyAudioAnalysisLibrary` this has been implemented in the `musicThumbnailing(x, Fs, shortTermSize=1.0, shortTermStep=0.5, thumbnailSize=10.0)` function from the `audioSegmentation.py`. The function uses the given wav file as an input music track and generates two thumbnails of `<thumbnailDuration>` length. It results are written in two wav files `<wavFileName>_thumb1.wav` and `<wavFileName>_thumb2.wav`

It uses `selfSimilarityMatrix()` that calculates the self-similarity matrix of an audio signal (also located in `audioSegmentation.py`)

Command-line use:
```
python audioAnalysis.py -thumbnail <wavFileName> <thumbnailDuration>
```

[Theodoros Giannakopoulos]: http://www.di.uoa.gr/~tyiannak
[Audio thumbnailing]: https://www.google.gr/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&sqi=2&ved=0CB4QFjAA&url=http%3A%2F%2Fmusic.ucsd.edu%2F~sdubnov%2FCATbox%2FReader%2FThumbnailingMM05.pdf&ei=pTX_U-i_K8S7ObiegMAP&usg=AFQjCNGT172T0VNB81IizPOyIYi3f58HJg&sig2=WAKASz6pvddafIMQlajXiA&bvm=bv.74035653,d.bGQ

