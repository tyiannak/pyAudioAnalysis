import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioSegmentation as aS
import numpy as np


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
    # 1) regression
    # 2) HMM?

    """

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
