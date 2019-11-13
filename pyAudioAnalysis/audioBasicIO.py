from __future__ import print_function
import os
import glob
import aifc
import numpy
import eyed3
import ntpath
import shutil
import numpy as np
from pydub import AudioSegment


def convertDirMP3ToWav(dirName, Fs, nC, useMp3TagsAsName = False):
    """
    This function converts the MP3 files stored in a folder to WAV. If required,
    the output names of the WAV files are based on MP3 tags, otherwise the same
    names are used.
    ARGUMENTS:
     - dirName:     the path of the folder where the MP3s are stored
     - Fs:          the sampling rate of the generated WAV files
     - nC:          the number of channels of the generated WAV files
     - useMp3TagsAsName:    True if the WAV filename is generated on MP3 tags
    """

    types = (dirName+os.sep+'*.mp3',)  # the tuple of file types
    filesToProcess = [] 

    for files in types:
        filesToProcess.extend(glob.glob(files))     

    for f in filesToProcess:
        audioFile = eyed3.load(f)               
        if useMp3TagsAsName and audioFile.tag != None:          
            artist = audioFile.tag.artist
            title = audioFile.tag.title
            if artist != None and title != None:
                if len(title) > 0 and len(artist) > 0:
                    wavFileName = ntpath.split(f)[0] + os.sep + \
                                  artist.replace(","," ") + " --- " + \
                                  title.replace(","," ") + ".wav"
                else:
                    wavFileName = f.replace(".mp3", ".wav")
            else:
                wavFileName = f.replace(".mp3", ".wav")
        else:
            wavFileName = f.replace(".mp3", ".wav")
        command = "avconv -i \"" + f + "\" -ar " + str(Fs) + " -ac " + \
                  str(nC) + "" + wavFileName + "\""
        print(command)
        os.system(command.decode('unicode_escape').encode('ascii', 'ignore')
                  .replace("\0", ""))


def convertFsDirWavToWav(dirName, Fs, nC):
    """
    This function converts the WAV files stored in a folder to WAV using a
    different sampling freq and number of channels.
    ARGUMENTS:
     - dirName:     the path of the folder where the WAVs are stored
     - Fs:          the sampling rate of the generated WAV files
     - nC:          the number of channesl of the generated WAV files
    """

    types = (dirName+os.sep+'*.wav',)  # the tuple of file types
    filesToProcess = []

    for files in types:
        filesToProcess.extend(glob.glob(files))     

    newDir = dirName + os.sep + "Fs" + str(Fs) + "_" + "NC"+str(nC)
    if os.path.exists(newDir) and newDir != ".":
        shutil.rmtree(newDir)   
    os.makedirs(newDir) 

    for f in filesToProcess:    
        _, wavFileName = ntpath.split(f)    
        command = "avconv -i \"" + f + "\" -ar " + str(Fs) + " -ac " + \
                  str(nC) + " \"" + newDir + os.sep + wavFileName + "\""
        print(command)
        os.system(command)


def read_audio_file(path):
    """
    This function returns a numpy array that stores the audio samples of a
    specified WAV of AIFF file
    """

    sampling_rate = -1
    signal = np.array([])
    extension = os.path.splitext(path)[1].lower()
    if extension in ['.aif', '.aiff']:
        sampling_rate, signal = read_aif(path)
    elif extension in [".mp3", ".wav", ".au", ".ogg"]:
        sampling_rate, signal = read_audio_generic(path)
    else:
        print(f"Error: unknown file type {extension}")

    if signal.ndim == 2 and signal.shape[1] == 1:
        signal = signal.flatten()

    return sampling_rate, signal


def read_aif(path):
    sampling_rate = -1
    signal = np.array([])
    try:
        with aifc.open(path, 'r') as s:
            nframes = s.getnframes()
            strsig = s.readframes(nframes)
            signal = numpy.fromstring(strsig, numpy.short).byteswap()
            sampling_rate = s.getframerate()
    except:
        print("Error: read aif file. (DECODING FAILED)")
    return sampling_rate, signal


def read_audio_generic(path):
    sampling_rate = -1
    signal = np.array([])
    try:
        audiofile = AudioSegment.from_file(path)
        data = np.array([])
        if audiofile.sample_width == 2:
            data = numpy.fromstring(audiofile._data, numpy.int16)
        elif audiofile.sample_width == 4:
            data = numpy.fromstring(audiofile._data, numpy.int32)

        if data.size > 0:
            sampling_rate = audiofile.frame_rate
            temp_signal = []
            for chn in list(range(audiofile.channels)):
                temp_signal.append(data[chn::audiofile.channels])
            signal = numpy.array(temp_signal).T
    except:
        print("Error: file not found or other I/O error. (DECODING FAILED)")
    return sampling_rate, signal


def stereo2mono(x):
    """
    This function converts the input signal
    (stored in a numpy array) to MONO (if it is STEREO)
    """
    if isinstance(x, int):
        return -1
    if x.ndim == 1:
        return x
    elif x.ndim == 2:
        if x.shape[1] == 1:
            return x.flatten()
        else:
            if x.shape[1] == 2:
                return (x[:, 1] / 2) + (x[:, 0] / 2)
            else:
                return -1

