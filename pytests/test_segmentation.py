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


    #TODOs 
    # 1) mid term classification (scottish using existing model in rep)