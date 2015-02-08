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
 * SKLEARN
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
 * `audioBasicIO.py`: this file implements some basic audio IO functionalities as well as file convertions 
 * `audioVisualization.py`: the purpose of this set of functions is to produce user-friendly and representative content visualizations

In the `data/` folder, a couple of audio sample files are provided, along with some trained SVM and kNN models for particular classification tasks (e.g. Speech vs Music, Musical Genre Classification, etc).

## Basic Functionalities

### Audio Feature Extraction
#### General
There are two stages in the audio feature extraction methodology: 
 * Short-term feature extraction: this is implemented in function `stFeatureExtraction()` of the `audioFeatureExtraction.py` file. It splits the input signal into short-term widnows (frames) and computes a number of features for each frame. This process leads to a sequence of short-term feature vectors for the whole signal.
 * Mid-term feature extraction: In many cases, the signal is represented by statistics on the extracted short-term feature sequences described above. Towards this end, function `mtFeatureExtraction()` from the `audioFeatureExtraction.py` file extracts a number of statistcs (e.g. mean and standard deviation) over each short-term feature sequence. 

TODO AN EXAMPLE

#### Single-file feature extraction - storing to file
The function used to generate short-term and mid-term features is `mtFeatureExtraction()` from the `audioFeatureExtraction.py` file. 
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


#### Spectrogram and Chromagram visualization
Functions `stSpectogram()` and `stChromagram()`  from the `audioFeatureExtraction.py` file can be used to generate the spectrogram and chromagram of an audio signal respectively. 

Command-line examples:
```
python audioAnalysis.py -fileSpectrogram data/doremi.wav
```

```
python audioAnalysis.py -fileChromagram data/doremi.wav
```

#### Beat extraction
Tempo induction is a rather important task in music information retrieval. This library provides a baseline method for estimating the beats per minute (BPM) rate of a music signal.
The beat rate estimation is implemented in function `beatExtraction()` of `audioFeatureExtraction.py` file. 
It accepts 2 arguments: (a) the short-term feature matrix and (b) the window step (in seconds).
Obviously, the `stFeatureExtraction` of the `audioFeatureExtraction.py` file is needed to extract the sequence of feature vectors before extracting the beat.

Command-line example:
```
python audioAnalysis.py  -beatExtraction data/beat/small.wav 1
```

The last argument should be 1 for visualizing the intermediate algorithmic stages (e.g. feature-specific local maxima detection, etc) and 0 otherwise (visualization can be very time consuming for >1 min signals). 

Note that the BPM feature is only applicable in the long-term analysis approach. 
Therefore, functions that perform long-term averaging on mid-term statistics (e.g. `dirWavFeatureExtraction()`) have also the choise to compute the BPM (and its confidence value) as features in the long-term feature representation. 


### Audio Classification
#### Train Segment Classifier From Data
A segment classification functionality is provided in the library. Towards this end, the `audioTrainTest.py` file implements two types of classifiers, namely the kNN and SVM methods. 
Below, we describe how to train a segment classifier from data (i.e. segments stored in WAV files, organized in directories that correspond to classes).

The function used to train a segment classifier model is `featureAndTrain(listOfDirs, mtWin, mtStep, stWin, stStep, classifierType, modelName, computeBEAT)` from `audioTrainTest.py`. The first argument is list of paths of directories. Each directory contains a signle audio class whose samples are stored in seperate WAV files. Then, the function takes the mid-term window size and step and the short-term window size and step respectively. 
The arguments `classifierType` and `modelName` are associated to the classifier type and name. The latest is also used as a name of the file where the model is stored for future use (see next sections on classification and segmentation).
Finally, the last argument is a boolean, set to `True` if the long-term beat-related features are to be calculated (e.g. for music classification tasks).
In addition, an ARFF file is also created (with the same name as the model), where the whole set of feature vectors and respective class labels are stored. 
Example:
```
from pyAudioAnalysis import audioTrainTest as aT
aT.featureAndTrain(["/home/tyiannak/Desktop/MusicGenre/Classical/","/home/tyiannak/Desktop/MusicGenre/Electronic/","/home/tyiannak/Desktop/MusicGenre/Jazz/"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmMusicGenre3", True)
aT.featureAndTrain(["/home/tyiannak/Desktop/MusicGenre/Classical/","/home/tyiannak/Desktop/MusicGenre/Electronic/","/home/tyiannak/Desktop/MusicGenre/Jazz/"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "knn", "knnMusicGenre3", True)
aT.featureAndTrain(["/home/tyiannak/Desktop/5Class/Silence/","/home/tyiannak/Desktop/5Class/SpeechMale/","/home/tyiannak/Desktop/5Class/SpeechFemale/","/home/tyiannak/Desktop/5Class/ObjectsOther/","/home/tyiannak/Desktop/5Class/Music/"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svm5Classes")
aT.featureAndTrain(["/home/tyiannak/Desktop/5Class/Silence/","/home/tyiannak/Desktop/5Class/SpeechMale/","/home/tyiannak/Desktop/5Class/SpeechFemale/","/home/tyiannak/Desktop/5Class/ObjectsOther/","/home/tyiannak/Desktop/5Class/Music/"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "knn", "knn5Classes")
```
Command-line use:
```
python audioAnalysis.py -trainClassifier <method(svm or knn)> <0 or 1 for beat-extraction enable> <directory 1> <directory 2> ... <directory N> <modelName>`. 
```
Examples:
```
python audioAnalysis.py -trainClassifier svm 0 /home/tyiannak/Desktop/SpeechMusic/music /home/tyiannak/Desktop/SpeechMusic/speech data/svmSM
```
```
python audioAnalysis.py -trainClassifier knn 0  ./data/SpeechMusic/speech ./data/SpeechMusic/music data/knnSM
```
```
python audioAnalysis.py -trainClassifier knn 1 /home/tyiannak/Desktop/MusicGenre/Classical/ /home/tyiannak/Desktop/MusicGenre/Electronic/ /home/tyiannak/Desktop/MusicGenre/Jazz/  data/knnMusicGenre3
```
```
python audioAnalysis.py -trainClassifier svm 1 /home/tyiannak/Desktop/MusicGenre/Classical/ /home/tyiannak/Desktop/MusicGenre/Electronic/ /home/tyiannak/Desktop/MusicGenre/Jazz/  data/svmMusicGenre3
```

#### Single File Classification
Function `fileClassification(inputFile, modelName, modelType)` from `audioTrainTest.py` file can be used to classify a single wav file based on an already trained segment classifier. 

Example:
```
from pyAudioAnalysis import audioTrainTest as aT
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
TODO COMMENTS HERE. Function `mtFileClassification` from `audioSegmentation.py`.

Example:
```
from pyAudioAnalysis import audioSegmentation as aS
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
#### Speaker Diarization
TODO 


#### Audio thumbnailing

[Audio thumbnailing] is an important application of music information retrieval that focuses on detecting instances of the most representative part of a music recording. In `pyAudioAnalysisLibrary` this has been implemented in the `musicThumbnailing(x, Fs, shortTermSize=1.0, shortTermStep=0.5, thumbnailSize=10.0)` function from the `audioSegmentation.py`. The function uses the given wav file as an input music track and generates two thumbnails of `<thumbnailDuration>` length. It results are written in two wav files `<wavFileName>_thumb1.wav` and `<wavFileName>_thumb2.wav`

It uses `selfSimilarityMatrix()` that calculates the self-similarity matrix of an audio signal (also located in `audioSegmentation.py`)

Command-line use:
```
python audioAnalysis.py -thumbnail <wavFileName> <thumbnailDuration>
```

### Data visualization 
The library provides the ability to visualize content similarities between audio recordings. 
Towards this end a [d3js] chordial representation has been adopted.
The core visualization functionality is provided in `audioVisualization.py` and in particular in function `visualizeFeaturesFolder()`. 
This function uses `dirWavFeatureExtraction()` to extract the long-term features for each of the WAV files contained in the provided folder. 
Then, a dimensionality reduction approach is performed using either the PCA or the LDA method. Since LDA is supervised, the required labels are taken from the 
subcategories of the input files (if available). These are provided through the respective filenames, using the string `---` as a seperator. 
For example, if folder contains the files:

```
Radiohead --- Lucky.wav
Radiohead --- Karma Police.wav
The Smashing Pumpkins --- Perfect.wav
The Smashing Pumpkins --- Rhinocerous.wav
```

then the labels `0, 0, 1, 1` are given to the features of the respetive filenames. 
In this context, the first part of the filename (if the seperator exists) defines the "group" (or the general category) of the respective recording.
In the example above the groups are `Radiohead` for the first two and `The Smashing Pumpkins` for the last two recordings. 
Note that during the convertion of MP3 to WAV (see function `convertDirMP3ToWav()`) the MP3 tags can be used in order to generate WAV filenames with an artist tag in the first half of their filename, just like the example above. 

As soon as the dimension of the feature space is reduced, a similarity matrix is computed (in the reduced space). 
Through thresholding this similarity matrix, a graph that illustrates the content similarities between the recordings' content is extracted.
This graph is represented using a [chordial diagram]. 
Different colors of the edges (recordings) represent different categories (artists in our case).
Links between edges correspond to content similarities.

Command-line example:

```
python audioAnalysis.py -featureVisualizationDir MusicData/
```

The above functionality results in 3 chordial diagrams: 
(a) one bases the compuation of the similarity matrix on the initial feature space 
(b) one that is based on the reduced space (either PCA or LDA) and 
(c) a chordial diagram of the groups' connections.
The three visualizations are respectivelly stored in three directories named `visualizationInitial_Chordial`, `visualization_Chordial`, and `visualizationGroup_Chordial`.


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

### Other Functionalities
#### Batch-convert Mp3 to Wav
Function `convertDirMP3ToWav(dirName, Fs, nC, useMp3TagsAsName = False)` converts all MP3 files of folder `dirName` to WAV files using the provided sampling rate (second argument) and number of channels (third argument). If the final argument (`useMp3TagsAsName`) is set to `True` then the output WAV files are named by the MP3-tags (artist and song title), otherwise the MP3 filename is used (with the .wav extension of course)

Command-line use example
```
python audioAnalysis.py -dirMp3toWAV data/beat/ 16000 1
```

[Theodoros Giannakopoulos]: http://www.di.uoa.gr/~tyiannak
[Audio thumbnailing]: https://www.google.gr/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&sqi=2&ved=0CB4QFjAA&url=http%3A%2F%2Fmusic.ucsd.edu%2F~sdubnov%2FCATbox%2FReader%2FThumbnailingMM05.pdf&ei=pTX_U-i_K8S7ObiegMAP&usg=AFQjCNGT172T0VNB81IizPOyIYi3f58HJg&sig2=WAKASz6pvddafIMQlajXiA&bvm=bv.74035653,d.bGQ
[d3js]: http://d3js.org/
[chordial diagram]: http://cgi.di.uoa.gr/~tyiannak/musicDemoVisualization/visualization_Chordial/similarities.html

