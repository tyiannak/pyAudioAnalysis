from __future__ import print_function
import os
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../"))
from pyAudioAnalysis import utilities
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
eps = 0.00000001

""" Time-domain audio features """


def beat_extraction(short_features, window_size, plot=False):
    """
    This function extracts an estimate of the beat rate for a musical signal.
    ARGUMENTS:
     - short_features:     a np array (n_feats x numOfShortTermWindows)
     - window_size:        window size in seconds
    RETURNS:
     - bpm:            estimates of beats per minute
     - ratio:          a confidence measure
    """

    # Features that are related to the beat tracking task:
    selected_features = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10,
                         11, 12, 13, 14, 15, 16, 17, 18]

    max_beat_time = int(round(2.0 / window_size))
    hist_all = np.zeros((max_beat_time,))
    # for each feature
    for ii, i in enumerate(selected_features):
        # dif threshold (3 x Mean of Difs)
        dif_threshold = 2.0 * (np.abs(short_features[i, 0:-1] -
                                      short_features[i, 1::])).mean()
        if dif_threshold <= 0:
            dif_threshold = 0.0000000000000001
        # detect local maxima
        [pos1, _] = utilities.peakdet(short_features[i, :], dif_threshold)
        position_diffs = []
        # compute histograms of local maxima changes
        for j in range(len(pos1)-1):
            position_diffs.append(pos1[j+1]-pos1[j])
        histogram_times, histogram_edges = \
            np.histogram(position_diffs, np.arange(0.5, max_beat_time + 1.5))
        hist_centers = (histogram_edges[0:-1] + histogram_edges[1::]) / 2.0
        histogram_times = \
            histogram_times.astype(float) / short_features.shape[1]
        hist_all += histogram_times
        if plot:
            plt.subplot(9, 2, ii + 1)
            plt.plot(short_features[i, :], 'k')
            for k in pos1:
                plt.plot(k, short_features[i, k], 'k*')
            f1 = plt.gca()
            f1.axes.get_xaxis().set_ticks([])
            f1.axes.get_yaxis().set_ticks([])

    if plot:
        plt.show(block=False)
        plt.figure()

    # Get beat as the argmax of the agregated histogram:
    max_indices = np.argmax(hist_all)
    bpms = 60 / (hist_centers * window_size)
    bpm = bpms[max_indices]
    # ... and the beat ratio:
    ratio = hist_all[max_indices] / (hist_all.sum() + eps)

    if plot:
        # filter out >500 beats from plotting:
        hist_all = hist_all[bpms < 500]
        bpms = bpms[bpms < 500]

        plt.plot(bpms, hist_all, 'k')
        plt.xlabel('Beats per minute')
        plt.ylabel('Freq Count')
        plt.show(block=True)

    return bpm, ratio


def mid_feature_extraction(signal, sampling_rate, mid_window, mid_step,
                           short_window, short_step):
    """
    Mid-term feature extraction
    """

    short_features, short_feature_names = \
        ShortTermFeatures.feature_extraction(signal, sampling_rate,
                                             short_window, short_step)

    n_stats = 2
    n_feats = len(short_features)
    #mid_window_ratio = int(round(mid_window / short_step))
    mid_window_ratio = round((mid_window -
                              (short_window - short_step)) / short_step)
    mt_step_ratio = int(round(mid_step / short_step))

    mid_features, mid_feature_names = [], []
    for i in range(n_stats * n_feats):
        mid_features.append([])
        mid_feature_names.append("")

    # for each of the short-term features:
    for i in range(n_feats):
        cur_position = 0
        num_short_features = len(short_features[i])
        mid_feature_names[i] = short_feature_names[i] + "_" + "mean"
        mid_feature_names[i + n_feats] = short_feature_names[i] + "_" + "std"

        while cur_position < num_short_features:
            end = cur_position + mid_window_ratio
            if end > num_short_features:
                end = num_short_features
            cur_st_feats = short_features[i][cur_position:end]

            mid_features[i].append(np.mean(cur_st_feats))
            mid_features[i + n_feats].append(np.std(cur_st_feats))
            cur_position += mt_step_ratio
    mid_features = np.array(mid_features)
    mid_features = np.nan_to_num(mid_features)
    return mid_features, short_features, mid_feature_names


""" Feature Extraction Wrappers
 - The first two feature extraction wrappers are used to extract 
   long-term averaged audio features for a list of WAV files stored in a 
   given category.
   It is important to note that, one single feature is extracted per WAV 
   file (not the whole sequence of feature vectors)

 """


def directory_feature_extraction(folder_path, mid_window, mid_step,
                                 short_window, short_step,
                                 compute_beat=True):
    """
    This function extracts the mid-term features of the WAVE files of a 
    particular folder.

    The resulting feature vector is extracted by long-term averaging the
    mid-term features.
    Therefore ONE FEATURE VECTOR is extracted for each WAV file.

    ARGUMENTS:
        - folder_path:        the path of the WAVE directory
        - mid_window, mid_step:    mid-term window and step (in seconds)
        - short_window, short_step:    short-term window and step (in seconds)
    """

    mid_term_features = np.array([])
    process_times = []

    types = ('*.wav', '*.aif',  '*.aiff', '*.mp3', '*.au', '*.ogg')
    wav_file_list = []
    for files in types:
        wav_file_list.extend(glob.glob(os.path.join(folder_path, files)))

    wav_file_list = sorted(wav_file_list)    
    wav_file_list2, mid_feature_names = [], []
    for i, file_path in enumerate(wav_file_list):
        print("Analyzing file {0:d} of {1:d}: {2:s}".format(i + 1,
                                                            len(wav_file_list),
                                                            file_path))
        if os.stat(file_path).st_size == 0:
            print("   (EMPTY FILE -- SKIPPING)")
            continue        
        sampling_rate, signal = audioBasicIO.read_audio_file(file_path)
        if sampling_rate == 0:
            continue        

        t1 = time.time()        
        signal = audioBasicIO.stereo_to_mono(signal)
        if signal.shape[0] < float(sampling_rate)/5:
            print("  (AUDIO FILE TOO SMALL - SKIPPING)")
            continue
        wav_file_list2.append(file_path)
        if compute_beat:
            mid_features, short_features, mid_feature_names = \
                mid_feature_extraction(signal, sampling_rate,
                                       round(mid_window * sampling_rate),
                                       round(mid_step * sampling_rate),
                                       round(sampling_rate * short_window),
                                       round(sampling_rate * short_step))
            beat, beat_conf = beat_extraction(short_features, short_step)
        else:
            mid_features, _, mid_feature_names = \
                mid_feature_extraction(signal, sampling_rate,
                                       round(mid_window * sampling_rate),
                                       round(mid_step * sampling_rate),
                                       round(sampling_rate * short_window),
                                       round(sampling_rate * short_step))

        mid_features = np.transpose(mid_features)
        mid_features = mid_features.mean(axis=0)
        # long term averaging of mid-term statistics
        if (not np.isnan(mid_features).any()) and \
                (not np.isinf(mid_features).any()):
            if compute_beat:
                mid_features = np.append(mid_features, beat)
                mid_features = np.append(mid_features, beat_conf)
                mid_feature_names += ["bpm","ratio"]
            if len(mid_term_features) == 0:
                # append feature vector
                mid_term_features = mid_features
            else:
                mid_term_features = np.vstack((mid_term_features, mid_features))
            t2 = time.time()
            duration = float(len(signal)) / sampling_rate
            process_times.append((t2 - t1) / duration)
    if len(process_times) > 0:
        print("Feature extraction complexity ratio: "
              "{0:.1f} x realtime".format((1.0 / 
                                           np.mean(np.array(process_times)))))
    return mid_term_features, wav_file_list2, mid_feature_names


def multiple_directory_feature_extraction(path_list, mid_window, mid_step,
                                          short_window, short_step,
                                          compute_beat=False):
    """
    Same as dirWavFeatureExtraction, but instead of a single dir it
    takes a list of paths as input and returns a list of feature matrices.
    EXAMPLE:
    [features, classNames] =
           a.dirsWavFeatureExtraction(['audioData/classSegmentsRec/noise',
                                       'audioData/classSegmentsRec/speech',
                                       'audioData/classSegmentsRec/brush-teeth',
                                       'audioData/classSegmentsRec/shower'], 1, 
                                       1, 0.02, 0.02);

    It can be used during the training process of a classification model ,
    in order to get feature matrices from various audio classes (each stored in
    a separate path)
    """

    # feature extraction for each class:
    features = []
    class_names = []
    file_names = []
    for i, d in enumerate(path_list):
        f, fn, feature_names = \
            directory_feature_extraction(d, mid_window, mid_step,
                                         short_window, short_step,
                                         compute_beat=compute_beat)
        if f.shape[0] > 0:
            # if at least one audio file has been found in the provided folder:
            features.append(f)
            file_names.append(fn)
            if d[-1] == os.sep:
                class_names.append(d.split(os.sep)[-2])
            else:
                class_names.append(d.split(os.sep)[-1])
    return features, class_names, file_names


def directory_feature_extraction_no_avg(folder_path, mid_window, mid_step,
                                        short_window, short_step):
    """
    This function extracts the mid-term features of the WAVE
    files of a particular folder without averaging each file.

    ARGUMENTS:
        - folder_path:          the path of the WAVE directory
        - mid_window, mid_step:    mid-term window and step (in seconds)
        - short_window, short_step:    short-term window and step (in seconds)
    RETURNS:
        - X:                A feature matrix
        - Y:                A matrix of file labels
        - filenames:
    """

    wav_file_list = []
    signal_idx = np.array([])
    mid_features = np.array([])
    types = ('*.wav', '*.aif',  '*.aiff', '*.ogg')
    for files in types:
        wav_file_list.extend(glob.glob(os.path.join(folder_path, files)))

    wav_file_list = sorted(wav_file_list)

    for i, file_path in enumerate(wav_file_list):
        sampling_rate, signal = audioBasicIO.read_audio_file(file_path)
        if sampling_rate == 0:
            continue
        signal = audioBasicIO.stereo_to_mono(signal)
        mid_feature_vector, _, _ = \
            mid_feature_extraction(signal, sampling_rate,
                                   round(mid_window * sampling_rate),
                                   round(mid_step * sampling_rate),
                                   round(sampling_rate * short_window),
                                   round(sampling_rate * short_step))

        mid_feature_vector = np.transpose(mid_feature_vector)
        if len(mid_features) == 0:                # append feature vector
            mid_features = mid_feature_vector
            signal_idx = np.zeros((mid_feature_vector.shape[0], ))
        else:
            mid_features = np.vstack((mid_features, mid_feature_vector))
            signal_idx = np.append(signal_idx, i *
                                   np.ones((mid_feature_vector.shape[0], )))

    return mid_features, signal_idx, wav_file_list


"""
The following two feature extraction wrappers extract features for given audio
files, however  NO LONG-TERM AVERAGING is performed. Therefore, the output for
each audio file is NOT A SINGLE FEATURE VECTOR but a whole feature matrix.

Also, another difference between the following two wrappers and the previous
is that they NO LONG-TERM AVERAGING IS PERFORMED. In other words, the WAV
files in these functions are not used as uniform samples that need to be
averaged but as sequences
"""


def mid_feature_extraction_to_file(file_path, mid_window, mid_step,
                                   short_window, short_step, output_file,
                                   store_short_features=False, store_csv=False,
                                   plot=False):
    """
    This function is used as a wrapper to:
    a) read the content of a WAV file
    b) perform mid-term feature extraction on that signal
    c) write the mid-term feature sequences to a np file
    d) optionally write contents to csv file as well
    e) optionally write short-term features in csv and np file
    """
    sampling_rate, signal = audioBasicIO.read_audio_file(file_path)
    signal = audioBasicIO.stereo_to_mono(signal)
    mid_features, short_features, _ = \
        mid_feature_extraction(signal, sampling_rate,
                               round(sampling_rate * mid_window),
                               round(sampling_rate * mid_step),
                               round(sampling_rate * short_window),
                               round(sampling_rate * short_step))
    if store_short_features:
        # save st features to np file
        np.save(output_file + "_st", short_features)
        if plot:
            print("Short-term np file: " + output_file + "_st.npy saved")
        if store_csv:
            # store st features to CSV file
            np.savetxt(output_file + "_st.csv", short_features.T, delimiter=",")
            if plot:
                print("Short-term CSV file: " + output_file + "_st.csv saved")

    # save mt features to np file
    np.save(output_file + "_mt", mid_features)
    if plot:
        print("Mid-term np file: " + output_file + "_mt.npy saved")
    if store_csv:
        np.savetxt(output_file + "_mt.csv", mid_features.T, delimiter=",")
        if plot:
            print("Mid-term CSV file: " + output_file + "_mt.csv saved")


def mid_feature_extraction_file_dir(folder_path, mid_window, mid_step,
                                    short_window, short_step,
                                    store_short_features=False, store_csv=False,
                                    plot=False):
    types = (folder_path + os.sep + '*.wav',)
    files_list = []
    for t in types:
        files_list.extend(glob.glob(t))
    for f in files_list:
        output_path = f
        mid_feature_extraction_to_file(f, mid_window, mid_step, short_window,
                                       short_step, output_path,
                                       store_short_features, store_csv, plot)
