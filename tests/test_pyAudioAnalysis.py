import argparse
import os
import sys
import pytest
sys.path.append('../')

import pyAudioAnalysis 
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures as STF 

@pytest.mark.parametrize('wav_file, plot', [ ('../pyAudioAnalysis/data/recording1.wav', True)])
def test_fileSpectrogramWrapper(wav_file, plot):
    if not os.path.isfile(wav_file):
        raise Exception("Input audio file not found!")
    [fs, x] = audioBasicIO.read_audio_file(wav_file)
    x = audioBasicIO.stereo_to_mono(x)
    win_length = 256 
    hop_length = win_length // 4
    specgram, TimeAxis, FreqAxis = STF.spectrogram(x, fs, win_length, hop_length, plot)

