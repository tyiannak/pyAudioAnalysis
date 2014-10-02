# pyAudioAnalysis: A Python Audio Analysis Library

## General
pyAudioAnalysis is a Python library for basic audio analysis tasks, including: feature extraction, classification, segmentation and visualization. 

*Author: [Theodoros Giannakopoulos]*

## Download
Type the following in your terminal:  
```
git clone https://github.com/tyiannak/pyAudioAnalysis.git
```

In order to be able to call the pyAudioAnalysis library from any path you need to add the folder that contains it in the ```~/.bashrc``` file. In particular, add a line as the follow in ```~/.bashrc```:

```
export PYTHONPATH=$PYTHONPATH:"/home/bla/bla"
```

(use the exact path where the ```pyAudioAnalysis``` folder is contained - without the ```pyAudioAnalysis``` name, e.g. if the library is contained in ```/home/tyiannak/Research/libraries/pyAudioAnalysis```, then use ```/home/tyiannak/Research/libraries``` in the ```bashrc``` file)

Then, you need to update the path details:

```
 source ~/.bashrc
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
 * sklearn
 ```
sudo apt-get install python-sklearn
```
 * Simplejson
```
sudo easy_install simplejson
```
 * eyeD3
```
sudo apt-get install python-eyed3
```



## General Structure
The library code is organized in 4 Python files. In particular:
 * `audioAnalysis.py`: this file implements the command-line interface of the basic functionalities of the library, along with some recording functionalities.
 * `audioFeatureExtraction.py`: this is where all audio feature extraction is implemented. In total, 21 short-term features are computed, while a mid-term windowing technique is also implemented, in order to extract statistics of audio features. 
 * `audioTrainTest.py`: this file implements the audio classification prodecures. It contains functions that can be used to train a Support Vector Machine or k-Nearest-Neighbour classifier. Also, wrapper functions and scripts are provided for general training, evaluating and feature normalization issues. 
 * `audioSegmentation.py`: this file implements audio segmentation functionalities, e.g. fixed-sized segment classification and segmentation, speaker diarization, etc. 

In the `data/` folder, a couple of audio sample files are provided, along with some trained SVM and kNN models for particular classification tasks (e.g. Speech vs Music, Musical Genre Classification, etc).

## Basic Functionalities

### Audio Feature Extraction
#### Single-file feature extraction - storing to file
The function used to generate short-term and mid-term features is `mtFeatureExtraction` from the `audioFeatureExtraction.py` file. 
This wrapping functionality also includes storing to CSV files and NUMPY files the short-term and mid-term feature matrices.
The command-line way to call this functionality is presented in the following example:

```
python audioAnalysis.py -featureExtractionFile data/speech_music_sample.wav 1.0 1.0 0.050 0.050
```
The result of this procedure are two comma-seperated files: `speech_music_sample.wav.csv` for the mid-term features and `speech_music_sample.wav_st.csv` for the short-term features. In each case, each feature sequence is stored in a seperate column, in other words, colums correspond to features and rows to time windows (short or long-term). Also, note that for the mid-term feature matrix, the number of features (columns) is two times higher than for the short-term analysis: this is due to the fact that the mid-term features are actually two statistics of the short-term features, namely the average value and the standard deviation. Also, note that in the mid-term feature matrix the first half of the values (in each time window) correspond to the average value, while the second half to the standard deviation of the respective short-term feature.
In the same way, the two feature matrices are stored in two numpy files (in this case: `speech_music_sample.wav.npy` and `speech_music_sample.wav_st.npy`). 
So in total four files are created during this process: two for mid-term features and two for short-term features. 

#### Feature extraction - storing to file for a sequence of WAV files stored in a given path
This functionality is the same as the one described above, however it works in a batch mode, i.e. it extracts four feature files for each WAV stored in the given folder.
Command-line example:
```
python audioAnalysis.py -featureExtractionDir data/ 1.0 1.0 0.050 0.050
```
The result of the above function is to generate feature files (2 CSVs and 2 NUMPY as described above), for each WAV file in the `data` folder.

Note: the feature extraction process described in the last two paragraphs, does not perform long-term averaging on the feature sequences, therefore a feature matrix is computed for each file (not a single feature vector).
See functions `dirWavFeatureExtraction()` and `dirsWavFeatureExtraction` for long-term averaging after the feature extraction process.

#### Data visualization 
TODO

### Audio Classification
#### Train Segment Classifier From Data
A segment classification functionality is provided in the library. Towards this end, the `audioTrainTest.py` file implements two types of classifiers, namely the kNN and SVM methods. 
Below, we describe how to train a segment classifier from data (i.e. segments stored in WAV files, organized in directories that correspond to classes).

The function used to train a segment classifier model is `featureAndTrain(listOfDirs, mtWin, mtStep, stWin, stStep, classifierType, modelName)` from `audioTrainTest.py`. The first argument is list of paths of directories. Each directory contains a signle audio class whose samples are stored in seperate WAV files. Then, the function takes the mid-term window size and step and the short-term window size and step respectively. 
Finally, the last two arguments are associated to the classifier type and name. The latest is also used as a name of the file where the model is stored for future use (see next sections on classification and segmentation).
In addition, an ARFF file is also created (with the same name as the model), where the whole set of feature vectors and respective class labels are stored. 
Example:
```
from audioAnalysisLibrary import audioTrainTest as aT
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
python audioAnalysis.py -trainClassifier knn ./data/SpeechMusic/speech ./data/SpeechMusic/music data/knnSM
```
```
python audioAnalysis.py -trainClassifier knn /home/tyiannak/Desktop/MusicGenre/Classical/ /home/tyiannak/Desktop/MusicGenre/Electronic/ /home/tyiannak/Desktop/MusicGenre/Jazz/  data/knnMusicGenre3
```
```
python audioAnalysis.py -trainClassifier svm /home/tyiannak/Desktop/MusicGenre/Classical/ /home/tyiannak/Desktop/MusicGenre/Electronic/ /home/tyiannak/Desktop/MusicGenre/Jazz/  data/svmMusicGenre3
```

#### Single File Classification
Function `fileClassification(inputFile, modelName, modelType)` from `audioTrainTest.py` file can be used to classify a single wav file based on an already trained segment classifier. 

Example:
```
from audioAnalysisLibrary import audioTrainTest as aT
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

#### Folder Classification
Classifies each WAV file found in the given folder and generates stdout resutls:
Command-line use examples:
```
python audioAnalysis.py -classifyFolder svm data/svmSM RecSegments/Speech/ 0 (only generates freq counts for each audio class)
python audioAnalysis.py -classifyFolder svm data/svmSM RecSegments/Speech/ 1 (also outputs the result of each singe WAV file)
```
### Audio Segmentation
#### File Segmentation & Classification
Function		`mtFileClassification` from `audioSegmentation.py`.

Example:
```
from audioAnalysisLibrary import audioSegmentation as aS
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

#### Audio thumbnailing

[Audio thumbnailing] is an important application of music information retrieval that focuses on detecting instances of the most representative part of a music recording. In `pyAudioAnalysisLibrary` this has been implemented in the `musicThumbnailing(x, Fs, shortTermSize=1.0, shortTermStep=0.5, thumbnailSize=10.0)` function from the `audioSegmentation.py`. The function uses the given wav file as an input music track and generates two thumbnails of `<thumbnailDuration>` length. It results are written in two wav files `<wavFileName>_thumb1.wav` and `<wavFileName>_thumb2.wav`

It uses `selfSimilarityMatrix()` that calculates the self-similarity matrix of an audio signal (also located in `audioSegmentation.py`)

Command-line use:
```
python audioAnalysis.py -thumbnail <wavFileName> <thumbnailDuration>
```
### Recording-related functionalities
Note: Some basic recording functionalities are also supported and demonstrated in `audioAnalysisRecordAlsa.py`. However, this requires the alsa-audio python library, only available in Linux, (`sudo apt-get install python-alsaaudio`) *

#### Record fix-sized audio segments
Function `recordAudioSegments(RecordPath, BLOCKSIZE)` from the `audioAnalysisRecordAlsa.py` file.

Command-line use example: 
```
python audioAnalysisRecordAlsa.py -recordSegments "rSpeech" 2.0
```

#### Realtime fix-sized segments classification
Function `recordAnalyzeAudio(duration, outputWavFile, midTermBufferSizeSec, modelName, modelType)` from the `audioAnalysisRecordAlsa.py` file. 

Command-line use example 
```
python audioAnalysisRecordAlsa.py -recordAndClassifySegments 20 out.wav knnRecNoiseActivity knn
```


[Theodoros Giannakopoulos]: http://www.di.uoa.gr/~tyiannak
[Audio thumbnailing]: https://www.google.gr/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&sqi=2&ved=0CB4QFjAA&url=http%3A%2F%2Fmusic.ucsd.edu%2F~sdubnov%2FCATbox%2FReader%2FThumbnailingMM05.pdf&ei=pTX_U-i_K8S7ObiegMAP&usg=AFQjCNGT172T0VNB81IizPOyIYi3f58HJg&sig2=WAKASz6pvddafIMQlajXiA&bvm=bv.74035653,d.bGQ

