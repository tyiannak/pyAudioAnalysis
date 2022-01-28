import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
from pyAudioAnalysis import MidTermFeatures
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioSegmentation as aS
import numpy as np

root_data_path = "/Users/tyiannak/ResearchData/Audio Dataset/pyAudioAnalysisData/"

def test_feature_extraction_short():
    [fs, x] = audioBasicIO.read_audio_file("test_data/1_sec_wav.wav")
    F, f_names = ShortTermFeatures.feature_extraction(x, fs, 
                                                      0.050 * fs, 0.050 * fs)
    assert F.shape[1] == 20, "Wrong number of mid-term windows"
    assert F.shape[0] == len(f_names), "Number of features and feature " \
                                       "names are not the same"


def test_feature_extraction_segment():
    print("Short-term feature extraction")
    [fs, x] = audioBasicIO.read_audio_file("test_data/5_sec_wav.wav")
    mt, st, mt_names = MidTermFeatures.mid_feature_extraction(x, fs, 
                                                              1 * fs, 
                                                              1 * fs, 
                                                              0.05 * fs, 
                                                              0.05 * fs)
    assert mt.shape[1] == 5, "Wrong number of short-term windows"
    assert mt.shape[0] == len(mt_names),  "Number of features and feature " \
                                          "names are not the same"


def test_speaker_diarization():
    labels, purity_cluster_m, purity_speaker_m = \
        aS.speaker_diarization("test_data/diarizationExample.wav", 
                                4, plot_res=False)
    assert purity_cluster_m > 0.9, "Diarization cluster purity is low"
    assert purity_speaker_m > 0.9, "Diarization speaker purity is low"


def test_train_and_evaluate_classifier():
    aT.extract_features_and_train(["test_data/3_class/music",
                                   "test_data/3_class/silence",
                                   "test_data/3_class/speech"], 
                                   1, 1, 0.05, 0.05, "svm_rbf", "temp")
    cm, thr_prre, pre, rec, thr_roc, fpr, tpr = \
        aT.evaluate_model_for_folders(["test_data/3_class/music",
                                       "test_data/3_class/silence",
                                       "test_data/3_class/speech"], 
                                      "temp", "svm_rbf", "music", False)
    acc = np.sum(np.diag(cm)) / np.sum(cm)
    assert acc > 0.9, "Low classification accuracy on training data"

    #TODOs 
    # 1) mid term classification (scottish using existing model in rep) 
    # 2) regression


    """
    print("\n\n\n * * * TEST 2 * * * \n\n\n")
    [Fs, x] = audioBasicIO.read_audio_file(root_data_path + "pyAudioAnalysis/data/doremi.wav")
    x = audioBasicIO.stereo_to_mono(x)
    specgram, TimeAxis, FreqAxis = ShortTermFeatures.spectrogram(x, Fs, round(Fs * 0.040), round(Fs * 0.040), True, True)

    print("\n\n\n * * * TEST 3 * * * \n\n\n")
    [Fs, x] = audioBasicIO.read_audio_file(root_data_path + "pyAudioAnalysis/data/doremi.wav")
    x = audioBasicIO.stereo_to_mono(x)
    specgram, TimeAxis, FreqAxis = ShortTermFeatures.chromagram(x, Fs, round(Fs * 0.040), round(Fs * 0.040), True)

    print("\n\n\n * * * TEST 4 * * * \n\n\n")
    aT.extract_features_and_train([root_data_path + "speakerAll/F1/", root_data_path + "speakerAll/F2/"], 1.0, 1.0, 0.2, 0.2, "svm", "temp", True)

    print("\n\n\n * * * TEST 5 * * * \n\n\n")
    [flagsInd, classesAll, acc, CM] = aS.mid_term_file_classification(root_data_path + "scottish.wav", root_data_path + "models/svm_rbf_sm", "svm_rbf", True, root_data_path + 'pyAudioAnalysis/data/scottish.segments')

    print("\n\n\n * * * TEST 6 * * * \n\n\n")
    aS.train_hmm_from_file(root_data_path + 'radioFinal/train/bbc4A.wav', root_data_path + 'radioFinal/train/bbc4A.segments', 'hmmTemp1', 1.0, 1.0)
    aS.train_hmm_from_directory(root_data_path + 'radioFinal/small', 'hmmTemp2', 1.0, 1.0)
    aS.hmm_segmentation(root_data_path + 'pyAudioAnalysis/data//scottish.wav', 'hmmTemp1', True, root_data_path + 'pyAudioAnalysis/data//scottish.segments')				# test 1
    aS.hmm_segmentation(root_data_path + 'pyAudioAnalysis/data//scottish.wav', 'hmmTemp2', True, root_data_path + 'pyAudioAnalysis/data//scottish.segments')				# test 2

    print("\n\n\n * * * TEST 7 * * * \n\n\n")
    aT.feature_extraction_train_regression(root_data_path + "pyAudioAnalysis/data/speechEmotion", 1, 1, 0.050, 0.050, "svm_rbf", "temp.mod", compute_beat=False)
    print(aT.file_regression(root_data_path + "pyAudioAnalysis/data/speechEmotion/01.wav", "temp.mod", "svm_rbf"))

    print("\n\n\n * * * TEST 8 * * * \n\n\n")
    aT.feature_extraction_train_regression(root_data_path + "pyAudioAnalysis/data/speechEmotion", 1, 1, 0.050, 0.050, "svm", "temp.mod", compute_beat=False)
    print(aT.file_regression(root_data_path + "pyAudioAnalysis/data/speechEmotion/01.wav", "temp.mod", "svm"))

    print("\n\n\n * * * TEST 9 * * * \n\n\n")
    aT.feature_extraction_train_regression(root_data_path + "pyAudioAnalysis/data/speechEmotion", 1, 1, 0.050, 0.050, "randomforest", "temp.mod", compute_beat=False)
    print(aT.file_regression(root_data_path + "pyAudioAnalysis/data/speechEmotion/01.wav", "temp.mod", "randomforest"))
    """
