import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioSegmentation as aS
import numpy as np


def test_speaker_diarization():
    labels, purity_cluster_m, purity_speaker_m = \
        aS.speaker_diarization("test_data/diarizationExample.wav", 
                                4, plot_res=False)
    assert purity_cluster_m > 0.9, "Diarization cluster purity is low"
    assert purity_speaker_m > 0.9, "Diarization speaker purity is low"


def test_mt_file_classification():
    labels, class_names, accuracy, cm = aS.mid_term_file_classification(
                                     "test_data/scottish.wav", 
                                     "test_data/svm_rbf_sm", "svm_rbf", False, 
                                     "test_data/scottish.segments")
    assert accuracy > 0.95, "Segment-level classification accuracy is low"
    