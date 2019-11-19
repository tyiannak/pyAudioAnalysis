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


def convert_dir_mp3_to_wav(audio_folder, sampling_rate, num_channels,
                           use_tags=False):
    """
    This function converts the MP3 files stored in a folder to WAV. If required,
    the output names of the WAV files are based on MP3 tags, otherwise the same
    names are used.
    ARGUMENTS:
     - audio_folder:    the path of the folder where the MP3s are stored
     - sampling_rate:   the sampling rate of the generated WAV files
     - num_channels:    the number of channels of the generated WAV files
     - use_tags:        True if the WAV filename is generated on MP3 tags
    """

    types = (audio_folder + os.sep + '*.mp3',)  # the tuple of file types
    files_list = []

    for files in types:
        files_list.extend(glob.glob(files))

    for f in files_list:
        audio_file = eyed3.load(f)
        if use_tags and audio_file.tag != None:
            artist = audio_file.tag.artist
            title = audio_file.tag.title
            if artist != None and title != None:
                if len(title) > 0 and len(artist) > 0:
                    filename = ntpath.split(f)[0] + os.sep + \
                                  artist.replace(","," ") + " --- " + \
                                  title.replace(","," ") + ".wav"
                else:
                    filename = f.replace(".mp3", ".wav")
            else:
                filename = f.replace(".mp3", ".wav")
        else:
            filename = f.replace(".mp3", ".wav")
        command = "avconv -i \"" + f + "\" -ar " + str(sampling_rate) + \
                  " -ac " + str(num_channels) + "" + filename + "\""
        print(command)
        os.system(command.decode('unicode_escape').encode('ascii', 'ignore')
                  .replace("\0", ""))


def convert_dir_fs_wav_to_wav(audio_folder, sampling_rate, num_channels):
    """
    This function converts the WAV files stored in a folder to WAV using a
    different sampling freq and number of channels.
    ARGUMENTS:
     - audio_folder:    the path of the folder where the WAVs are stored
     - sampling_rate:   the sampling rate of the generated WAV files
     - num_channels:    the number of channesl of the generated WAV files
    """

    types = (audio_folder + os.sep + '*.wav',)  # the tuple of file types

    files_list = []
    for files in types:
        files_list.extend(glob.glob(files))

    output_folder = audio_folder + os.sep + "Fs" + str(sampling_rate) + \
                    "_" + "NC" + str(num_channels)
    if os.path.exists(output_folder) and output_folder != ".":
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    for f in files_list:
        _, filename = ntpath.split(f)
        command = "avconv -i \"" + f + "\" -ar " + str(sampling_rate) + \
                  " -ac " + str(num_channels) + " \"" + output_folder + \
                  os.sep + filename + "\""
        print(command)
        os.system(command)


def read_audio_file(path):
    """
    This function returns a numpy array that stores the audio samples of a
    specified WAV of AIFF file
    """

    sampling_rate = 0
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
    """
    Read audio file with .aif extension
    """
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
    """
    Function to read audio files with the following extensions
    [".mp3", ".wav", ".au", ".ogg"]
    """
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


def stereo_to_mono(signal):
    """
    This function converts the input signal
    (stored in a numpy array) to MONO (if it is STEREO)
    """

    if signal.ndim == 2:
        if signal.shape[1] == 1:
            signal = signal.flatten()
        else:
            if signal.shape[1] == 2:
                signal = (signal[:, 1] / 2) + (signal[:, 0] / 2)
    return signal
