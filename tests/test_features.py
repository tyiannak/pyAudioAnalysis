import argparse
import pytest
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../')

import pyAudioAnalysis 
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures as STF 


@pytest.mark.parametrize('wav_file, plot', [ ('../pyAudioAnalysis/data/recording1.wav', True)])
def test_shortTermFeatures(wav_file, plot):
    [fs, data] = audioBasicIO.read_audio_file(wav_file)
    print(f'FS={fs} win={0.050*fs} step={0.025*fs}')
    F,f = STF.feature_extraction_lengthwise(data, fs, 0.050*fs, 0.025*fs);

    if plot:
        fig = plt.figure(figsize=(12, 6)) 
        ax1 = fig.subplots() 
        ax2 = ax1.twinx() 
        ax3 = ax2.twinx() 

        ax1.plot(F[1,:], color='red', label=f[1])
        ax2.plot(F[0,:], color='green', label=f[0])
        ax3.plot(data, color='blue', label='data', alpha=0.5)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax3.set_xlabel('time (s)')
        ax3.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc=0)
        ax1.axis('off')
        ax2.axis('off')
        #fig.savefig('recording1_shortTermFeatures.png', dpi=200)
        plt.show()

    return fig
