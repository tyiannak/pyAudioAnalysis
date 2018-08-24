from __future__ import print_function
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioSegmentation as aS
import matplotlib.pyplot as plt

root_data_path = "/Users/tyiannak/ResearchData/Audio Dataset/pyAudioAnalysisData/"

print("\n\n\n * * * TEST 1 * * * \n\n\n")
[Fs, x] = audioBasicIO.readAudioFile(root_data_path + "pyAudioAnalysis/data/count.wav");
F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs);
plt.subplot(2,1,1); plt.plot(F[0,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[0]);
plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[1]); plt.show()

print("\n\n\n * * * TEST 2 * * * \n\n\n")
[Fs, x] = audioBasicIO.readAudioFile(root_data_path + "pyAudioAnalysis/data/doremi.wav")
x = audioBasicIO.stereo2mono(x)
specgram, TimeAxis, FreqAxis = audioFeatureExtraction.stSpectogram(x, Fs, round(Fs * 0.040), round(Fs * 0.040), True)

print("\n\n\n * * * TEST 3 * * * \n\n\n")
[Fs, x] = audioBasicIO.readAudioFile(root_data_path + "pyAudioAnalysis/data/doremi.wav")
x = audioBasicIO.stereo2mono(x)
specgram, TimeAxis, FreqAxis = audioFeatureExtraction.stChromagram(x, Fs, round(Fs * 0.040), round(Fs * 0.040), True)

print("\n\n\n * * * TEST 4 * * * \n\n\n")
aT.featureAndTrain([root_data_path +"SM/speech",root_data_path + "SM/music"], 1.0, 1.0, 0.2, 0.2, "svm", "temp", True)

print("\n\n\n * * * TEST 5 * * * \n\n\n")
[flagsInd, classesAll, acc, CM] = aS.mtFileClassification(root_data_path + "pyAudioAnalysis/data//scottish.wav", root_data_path + "pyAudioAnalysis/data/svmSM", "svm", True, root_data_path + 'pyAudioAnalysis/data/scottish.segments')

print("\n\n\n * * * TEST 6 * * * \n\n\n")
aS.trainHMM_fromFile(root_data_path + 'radioFinal/train/bbc4A.wav', root_data_path + 'radioFinal/train/bbc4A.segments', 'hmmTemp1', 1.0, 1.0)	
aS.trainHMM_fromDir(root_data_path + 'radioFinal/small', 'hmmTemp2', 1.0, 1.0)
aS.hmmSegmentation(root_data_path + 'pyAudioAnalysis/data//scottish.wav', 'hmmTemp1', True, root_data_path + 'pyAudioAnalysis/data//scottish.segments')				# test 1
aS.hmmSegmentation(root_data_path + 'pyAudioAnalysis/data//scottish.wav', 'hmmTemp2', True, root_data_path + 'pyAudioAnalysis/data//scottish.segments')				# test 2

print("\n\n\n * * * TEST 7 * * * \n\n\n")
aT.featureAndTrainRegression(root_data_path + "pyAudioAnalysis/data/speechEmotion", 1, 1, 0.050, 0.050, "svm_rbf", "temp.mod", compute_beat=False)
print(aT.fileRegression(root_data_path + "pyAudioAnalysis/data/speechEmotion/01.wav", "temp.mod", "svm_rbf"))

print("\n\n\n * * * TEST 8 * * * \n\n\n")
aT.featureAndTrainRegression(root_data_path + "pyAudioAnalysis/data/speechEmotion", 1, 1, 0.050, 0.050, "svm", "temp.mod", compute_beat=False)
print(aT.fileRegression(root_data_path + "pyAudioAnalysis/data/speechEmotion/01.wav", "temp.mod", "svm"))

print("\n\n\n * * * TEST 9 * * * \n\n\n")
aT.featureAndTrainRegression(root_data_path + "pyAudioAnalysis/data/speechEmotion", 1, 1, 0.050, 0.050, "randomforest", "temp.mod", compute_beat=False)
print(aT.fileRegression(root_data_path + "pyAudioAnalysis/data/speechEmotion/01.wav", "temp.mod", "randomforest"))

