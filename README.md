# pyAudioAnalysis: A Python Audio Analysis Library

## General
pyAudioAnalysis is a python library for basic audio analysis tasks, including: feature extraction, classification, segmentation and visualization. 
+++

## Download
Type the following in your terminal:  `git clone https://github.com/tyiannak/pyAudioAnalysis.git`

## Dependencies

Check out this neat program I wrote:

```
x = 0
x = 2 + 2
what is x
```


 * MLPY:

```
wget http://sourceforge.net/projects/mlpy/files/mlpy%203.5.0/mlpy-3.5.0.tar.gz
tar xvf mlpy-3.5.0.tar.gz
cd mlpy-3.5.0
sudo python setup.py install
```

 * NUMPY:		sudo apt-get install python-numpy
 * MATPLOTLIB:	sudo apt-get install python-matplotlib
 * SCIPY:		sudo apt-get install python-scipy
 * GSL: 		sudo apt-get install libgsl0-dev
 * AlsaAudio: 	sudo apt-get install python-alsaaudio

## Basic Functionalities

Record fix-sized audio segments
	Function: 		recordAudioSegments(RecordPath, BLOCKSIZE) of audioAnalysis.py
	Command-line use: 	python audioAnalysis.py -recordSegments "rSpeech" 2.0

Realtime fix-sized segments classification
	Function: 		recordAnalyzeAudio(duration, outputWavFile, midTermBufferSizeSec, modelName, modelType)
	Command-line use:	python audioAnalysis.py -recordAndClassifySegments 20 out.wav knnRecNoiseActivity knn

Train Segment Classifier From Data
A segment classification functionality is provided in the library. Towards this end, the audioTrainTest.py file implements two types of classifiers, namelly the kNN and SVM methods.
Below, we describe how to train a segment classifier from data (i.e. segments stored in WAV files, organized in directories that correspond to classes).
	Function: 		featureAndTrain() from audioTrainTest.py
				Example:
				import audioTrainTest as aT
				aT.featureAndTrain(["/home/tyiannak/Desktop/MusicGenre/Classical/","/home/tyiannak/Desktop/MusicGenre/Electronic/","/home/tyiannak/Desktop/MusicGenre/Jazz/"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmMusicGenre3")
				aT.featureAndTrain(["/home/tyiannak/Desktop/MusicGenre/Classical/","/home/tyiannak/Desktop/MusicGenre/Electronic/","/home/tyiannak/Desktop/MusicGenre/Jazz/"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "knn", "knnMusicGenre3")
				aT.featureAndTrain(["/home/tyiannak/Desktop/5Class/Silence/","/home/tyiannak/Desktop/5Class/SpeechMale/","/home/tyiannak/Desktop/5Class/SpeechFemale/","/home/tyiannak/Desktop/5Class/ObjectsOther/","/home/tyiannak/Desktop/5Class/Music/"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svm5Classes")
				aT.featureAndTrain(["/home/tyiannak/Desktop/5Class/Silence/","/home/tyiannak/Desktop/5Class/SpeechMale/","/home/tyiannak/Desktop/5Class/SpeechFemale/","/home/tyiannak/Desktop/5Class/ObjectsOther/","/home/tyiannak/Desktop/5Class/Music/"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "knn", "knn5Classes")

	Command-line use:	python audioAnalysis.py -trainClassifier <method(svm or knn)> <directory 1> <directory 2> ... <directory N> <modelName>
				Examples
				python audioAnalysis.py -trainClassifier svm /home/tyiannak/Desktop/SpeechMusic/music /home/tyiannak/Desktop/SpeechMusic/speech svmSM
				python audioAnalysis.py -trainClassifier knn /home/tyiannak/Desktop/ /home/tyiannak/Desktop/SpeechMusic/speech knnSM
				python audioAnalysis.py -trainClassifier knn /home/tyiannak/Desktop/MusicGenre/Classical/ /home/tyiannak/Desktop/MusicGenre/Electronic/ /home/tyiannak/Desktop/MusicGenre/Jazz/  knnMusicGenre3
				python audioAnalysis.py -trainClassifier svm /home/tyiannak/Desktop/MusicGenre/Classical/ /home/tyiannak/Desktop/MusicGenre/Electronic/ /home/tyiannak/Desktop/MusicGenre/Jazz/  svmMusicGenre3

Single File Classification
	Function:		fileClassification from audioTrainTest library
				Example:
				import audioTrainTest as aT
				aT.fileClassification("TrueFaith.wav", "svmMusicGenre3","svm")
	Command-line use:	python audioAnalysis.py -classifyFile <method(svm or knn)> <modelName> <fileName>
				Examples:
				python audioAnalysis.py -classifyFile knn knnSM TrueFaith.wav
				python audioAnalysis.py -classifyFile knn knnMusicGenre3 TrueFaith.wav
				python audioAnalysis.py -classifyFile svm svmMusicGenre3 TrueFaith.wav

Folder Classification
 Classifies each WAV file found in the given folder and generates stdout resutls:
	Command-line use:	Examples:
				python audioAnalysis.py -classifyFolder svm svmSM RecSegments/Speech/ 0 (only generates freq counts for each audio class)
				python audioAnalysis.py -classifyFolder svm svmSM RecSegments/Speech/ 1 (also outputs the result of each singe WAV file)



File Segmentation & Classification
	Function		mtFileClassification (audioSegmentation.py)
				Example:
				import audioSegmentation as aS
				[segs, classes] = aS.mtFileClassification("data/speech_music_sample.wav", "data/svmSM", "svm", True)
	Command-line use:	python audioAnalysis.py -segmentClassifyFile <method(svm or knn)> <modelName> <fileName>
				Example:				
				python audioAnalysis.py -segmentClassifyFile svm data/svmSM data/speech_music_sample.wav 
Audio thumbnailing
Uses <wavFileName> as input music track and generates two thumbnails of <thumbnailDuration> length.
Also, results are written in two wav files <wavFileName>_thumb1.wav and <wavFileName>_thumb2.wav
 	Function: 		musicThumbnailing() in audioSegmentation.py Also uses selfSimilarityMatrix() in audioSegmentation.py
	Command-line use:	python audioAnalysis.py -thumbnail <wavFileName> <thumbnailDuration>


