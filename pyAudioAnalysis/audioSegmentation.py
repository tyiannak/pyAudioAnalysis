from __future__ import print_function
import os
import csv
import glob
import scipy
import sklearn
import numpy as np
import hmmlearn.hmm
import sklearn.cluster
import pickle as cpickle
import matplotlib.pyplot as plt
from scipy.spatial import distance
import sklearn.discriminant_analysis
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioTrainTest as at
from pyAudioAnalysis import MidTermFeatures as mtf
from pyAudioAnalysis import ShortTermFeatures as stf

""" General utility functions """


def smooth_moving_avg(signal, window=11):
    window = int(window)
    if signal.ndim != 1:
        raise ValueError("")
    if signal.size < window:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window < 3:
        return signal
    s = np.r_[2 * signal[0] - signal[window - 1::-1],
              signal, 2 * signal[-1] - signal[-1:-window:-1]]
    w = np.ones(window, 'd')
    y = np.convolve(w/w.sum(), s, mode='same')
    return y[window:-window + 1]


def self_similarity_matrix(feature_vectors):
    """
    This function computes the self-similarity matrix for a sequence
    of feature vectors.
    ARGUMENTS:
     - feature_vectors:    a np matrix (nDims x nVectors) whose i-th column
                           corresponds to the i-th feature vector

    RETURNS:
     - sim_matrix:         the self-similarity matrix (nVectors x nVectors)
    """
    norm_feature_vectors, mean, std = at.normalizeFeatures([feature_vectors.T])
    norm_feature_vectors = norm_feature_vectors[0].T
    sim_matrix = 1.0 - distance.squareform(
        distance.pdist(norm_feature_vectors.T, 'cosine'))
    return sim_matrix


def labels_to_segments(labels, window):
    """
    ARGUMENTS:
     - labels:     a sequence of class labels (per time window)
     - window:     window duration (in seconds)

    RETURNS:
     - segments:   a sequence of segment's limits: segs[i,0] is start and
                   segs[i,1] are start and end point of segment i
     - classes:    a sequence of class flags: class[i] is the class ID of
                   the i-th segment
    """

    num_segs = 0
    index = 0
    classes = []
    segment_list = []
    cur_label = labels[index]
    while index < len(labels) - 1:
        previous_value = cur_label
        while True:
            index += 1
            compare_flag = labels[index]
            if (compare_flag != cur_label) | (index == len(labels) - 1):
                num_segs += 1
                cur_label = labels[index]
                segment_list.append((index * window))
                classes.append(previous_value)
                break
    segments = np.zeros((len(segment_list), 2))

    for i in range(len(segment_list)):
        if i > 0:
            segments[i, 0] = segment_list[i-1]
        segments[i, 1] = segment_list[i]
    return segments, classes


def segments_to_labels(start_times, end_times, labels, window):
    """
    This function converts segment endpoints and respective segment
    labels to fix-sized class labels.
    ARGUMENTS:
     - start_times:  segment start points (in seconds)
     - end_times:    segment endpoints (in seconds)
     - labels:       segment labels
     - window:      fix-sized window (in seconds)
    RETURNS:
     - flags:    np array of class indices
     - class_names:    list of classnames (strings)
    """
    flags = []
    class_names = list(set(labels))
    index = window / 2.0
    while index < end_times[-1]:
        for i in range(len(start_times)):
            if start_times[i] < index <= end_times[i]:
                break
        flags.append(class_names.index(labels[i]))
        index += window
    return np.array(flags), class_names


def compute_metrics(confusion_matrix, class_names):
    """
    This function computes the precision, recall and f1 measures,
    given a confusion matrix
    """
    f1 = []
    recall = []
    precision = []
    n_classes = confusion_matrix.shape[0]
    if len(class_names) != n_classes:
        print("Error in computePreRec! Confusion matrix and class_names "
              "list must be of the same size!")
    else:
        for i, c in enumerate(class_names):
            precision.append(confusion_matrix[i, i] /
                             np.sum(confusion_matrix[:, i]))
            recall.append(confusion_matrix[i, i] /
                          np.sum(confusion_matrix[i, :]))
            f1.append(2 * precision[-1] * recall[-1] /
                      (precision[-1] + recall[-1]))
    return recall, precision, f1


def read_segmentation_gt(gt_file):
    """
    This function reads a segmentation ground truth file,
    following a simple CSV format with the following columns:
    <segment start>,<segment end>,<class label>

    ARGUMENTS:
     - gt_file:       the path of the CSV segment file
    RETURNS:
     - seg_start:     a np array of segments' start positions
     - seg_end:       a np array of segments' ending positions
     - seg_label:     a list of respective class labels (strings)
    """
    with open(gt_file, 'rt') as f_handle:
        reader = csv.reader(f_handle, delimiter=',')
        start_times = []
        end_times = []
        labels = []
        for row in reader:
            if len(row) == 3:
                start_times.append(float(row[0]))
                end_times.append(float(row[1]))
                labels.append((row[2]))
    return np.array(start_times), np.array(end_times), labels


def plot_segmentation_results(flags_ind, flags_ind_gt, class_names, mt_step,
                              evaluate_only=False):
    """
    This function plots statistics on the classification-segmentation results 
    produced either by the fix-sized supervised method or the HMM method.
    It also computes the overall accuracy achieved by the respective method 
    if ground-truth is available.
    """
    
    flags = [class_names[int(f)] for f in flags_ind]
    segments, classes = labels_to_segments(flags, mt_step)
    min_len = min(flags_ind.shape[0], flags_ind_gt.shape[0])    
    if min_len > 0:
        accuracy = np.sum(flags_ind[0:min_len] ==
                          flags_ind_gt[0:min_len]) / float(min_len)
    else:
        accuracy = -1

    if not evaluate_only:
        duration = segments[-1, 1]
        s_percentages = np.zeros((len(class_names), ))
        percentages = np.zeros((len(class_names), ))
        av_durations = np.zeros((len(class_names), ))

        for i_seg in range(segments.shape[0]):
            s_percentages[class_names.index(classes[i_seg])] += \
                (segments[i_seg, 1]-segments[i_seg, 0])

        for i in range(s_percentages.shape[0]):
            percentages[i] = 100.0 * s_percentages[i] / duration
            class_sum = sum(1 for c in classes if c == class_names[i])
            if class_sum > 0:
                av_durations[i] = s_percentages[i] / class_sum
            else:
                av_durations[i] = 0.0

        for i in range(percentages.shape[0]):
            print(class_names[i], percentages[i], av_durations[i])

        font = {'size': 10}
        plt.rc('font', **font)

        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.set_yticks(np.array(range(len(class_names))))
        ax1.axis((0, duration, -1, len(class_names)))
        ax1.set_yticklabels(class_names)
        ax1.plot(np.array(range(len(flags_ind))) * mt_step +
                 mt_step / 2.0, flags_ind)
        if flags_ind_gt.shape[0] > 0:
            ax1.plot(np.array(range(len(flags_ind_gt))) * mt_step +
                     mt_step / 2.0, flags_ind_gt + 0.05, '--r')
        plt.xlabel("time (seconds)")
        if accuracy >= 0:
            plt.title('Accuracy = {0:.1f}%'.format(100.0 * accuracy))

        ax2 = fig.add_subplot(223)
        plt.title("Classes percentage durations")
        ax2.axis((0, len(class_names) + 1, 0, 100))
        ax2.set_xticks(np.array(range(len(class_names) + 1)))
        ax2.set_xticklabels([" "] + class_names)
        print(np.array(range(len(class_names))), percentages)
        ax2.bar(np.array(range(len(class_names))) + 0.5, percentages)

        ax3 = fig.add_subplot(224)
        plt.title("Segment average duration per class")
        ax3.axis((0, len(class_names)+1, 0, av_durations.max()))
        ax3.set_xticks(np.array(range(len(class_names) + 1)))
        ax3.set_xticklabels([" "] + class_names)
        ax3.bar(np.array(range(len(class_names))) + 0.5, av_durations)
        fig.tight_layout()
        plt.show()
    return accuracy


def evaluate_speaker_diarization(labels, labels_gt):

    min_len = min(labels.shape[0], labels_gt.shape[0])
    labels = labels[0:min_len]
    labels_gt = labels_gt[0:min_len]

    unique_flags = np.unique(labels)
    unique_flags_gt = np.unique(labels_gt)

    # compute contigency table:
    contigency_matrix = np.zeros((unique_flags.shape[0],
                                  unique_flags_gt.shape[0]))
    for i in range(min_len):
        contigency_matrix[int(np.nonzero(unique_flags == labels[i])[0]),
                int(np.nonzero(unique_flags_gt == labels_gt[i])[0])] += 1.0

    columns, rows = contigency_matrix.shape
    row_sum = np.sum(contigency_matrix, axis=0)
    column_sum = np.sum(contigency_matrix, axis=1)
    matrix_sum = np.sum(contigency_matrix)

    purity_clust = np.zeros((columns, ))
    purity_speak = np.zeros((rows, ))
    # compute cluster purity:
    for i in range(columns):
        purity_clust[i] = np.max((contigency_matrix[i, :])) / (column_sum[i])

    for j in range(rows):
        purity_speak[j] = np.max((contigency_matrix[:, j])) / (row_sum[j])

    purity_cluster_m = np.sum(purity_clust * column_sum) / matrix_sum
    purity_speaker_m = np.sum(purity_speak * row_sum) / matrix_sum

    return purity_cluster_m, purity_speaker_m


def train_hmm_compute_statistics(features, labels):
    """
    This function computes the statistics used to train
    an HMM joint segmentation-classification model
    using a sequence of sequential features and respective labels

    ARGUMENTS:
     - features:  a np matrix of feature vectors (numOfDimensions x n_wins)
     - labels:    a np array of class indices (n_wins x 1)
    RETURNS:
     - class_priors:            matrix of prior class probabilities
                                (n_classes x 1)
     - transmutation_matrix:    transition matrix (n_classes x n_classes)
     - means:                   means matrix (numOfDimensions x 1)
     - cov:                     deviation matrix (numOfDimensions x 1)
    """
    unique_labels = np.unique(labels)
    n_comps = len(unique_labels)

    n_feats = features.shape[0]

    if features.shape[1] < labels.shape[0]:
        print("trainHMM warning: number of short-term feature vectors "
              "must be greater or equal to the labels length!")
        labels = labels[0:features.shape[1]]

    # compute prior probabilities:
    class_priors = np.zeros((n_comps,))
    for i, u_label in enumerate(unique_labels):
        class_priors[i] = np.count_nonzero(labels == u_label)
    # normalize prior probabilities
    class_priors = class_priors / class_priors.sum()

    # compute transition matrix:
    transmutation_matrix = np.zeros((n_comps, n_comps))
    for i in range(labels.shape[0]-1):
        transmutation_matrix[int(labels[i]), int(labels[i + 1])] += 1
    # normalize rows of transition matrix:
    for i in range(n_comps):
        transmutation_matrix[i, :] /= transmutation_matrix[i, :].sum()

    means = np.zeros((n_comps, n_feats))
    for i in range(n_comps):
        means[i, :] = \
            np.array(features[:,
                     np.nonzero(labels == unique_labels[i])[0]].mean(axis=1))

    cov = np.zeros((n_comps, n_feats))
    for i in range(n_comps):
        """
        cov[i, :, :] = np.cov(features[:, np.nonzero(labels == u_labels[i])[0]])
        """
        # use line above if HMM using full gaussian distributions are to be used
        cov[i, :] = np.std(features[:,
                           np.nonzero(labels == unique_labels[i])[0]],
                           axis=1)

    return class_priors, transmutation_matrix, means, cov


def train_hmm_from_file(wav_file, gt_file, hmm_model_name, mt_win, mt_step):
    """
    This function trains a HMM model for segmentation-classification
    using a single annotated audio file
    ARGUMENTS:
     - wav_file:        the path of the audio filename
     - gt_file:         the path of the ground truth filename
                       (a csv file of the form <segment start in seconds>,
                       <segment end in seconds>,<segment label> in each row
     - hmm_model_name:   the name of the HMM model to be stored
     - mt_win:          mid-term window size
     - mt_step:         mid-term window step
    RETURNS:
     - hmm:            an object to the resulting HMM
     - class_names:     a list of class_names

    After training, hmm, class_names, along with the mt_win and mt_step
    values are stored in the hmm_model_name file
    """

    seg_start, seg_end, seg_labs = read_segmentation_gt(gt_file)
    flags, class_names = segments_to_labels(seg_start, seg_end, seg_labs, mt_step)
    sampling_rate, signal = audioBasicIO.read_audio_file(wav_file)
    features, _, _ = \
        mtf.mid_feature_extraction(signal, sampling_rate,
                                   mt_win * sampling_rate,
                                   mt_step * sampling_rate,
                                   round(sampling_rate * 0.050),
                                   round(sampling_rate * 0.050))
    class_priors, transumation_matrix, means, cov = \
        train_hmm_compute_statistics(features, flags)
    hmm = hmmlearn.hmm.GaussianHMM(class_priors.shape[0], "diag")

    hmm.covars_ = cov
    hmm.means_ = means
    hmm.startprob_ = class_priors
    hmm.transmat_ = transumation_matrix

    with open(hmm_model_name, "wb") as f_handle:
        cpickle.dump(hmm, f_handle, protocol=cpickle.HIGHEST_PROTOCOL)
        cpickle.dump(class_names, f_handle, protocol=cpickle.HIGHEST_PROTOCOL)
        cpickle.dump(mt_win, f_handle, protocol=cpickle.HIGHEST_PROTOCOL)
        cpickle.dump(mt_step, f_handle, protocol=cpickle.HIGHEST_PROTOCOL)

    return hmm, class_names


def train_hmm_from_directory(folder_path, hmm_model_name, mt_win, mt_step):
    """
    This function trains a HMM model for segmentation-classification using
    a where WAV files and .segment (ground-truth files) are stored
    ARGUMENTS:
     - folder_path:     the path of the data diretory
     - hmm_model_name:  the name of the HMM model to be stored
     - mt_win:          mid-term window size
     - mt_step:         mid-term window step
    RETURNS:
     - hmm:            an object to the resulting HMM
     - class_names:    a list of class_names

    After training, hmm, class_names, along with the mt_win
    and mt_step values are stored in the hmm_model_name file
    """

    flags_all = np.array([])
    classes_all = []
    for i, f in enumerate(glob.glob(folder_path + os.sep + '*.wav')):
        # for each WAV file
        wav_file = f
        gt_file = f.replace('.wav', '.segments')
        if os.path.isfile(gt_file):
            seg_start, seg_end, seg_labs = read_segmentation_gt(gt_file)
            flags, class_names = \
                segments_to_labels(seg_start, seg_end, seg_labs, mt_step)
            for c in class_names:
                # update class names:
                if c not in classes_all:
                    classes_all.append(c)
            sampling_rate, signal = audioBasicIO.read_audio_file(wav_file)
            feature_vector, _, _ = \
                mtf.mid_feature_extraction(signal, sampling_rate,
                                           mt_win * sampling_rate,
                                           mt_step * sampling_rate,
                                           round(sampling_rate * 0.050),
                                           round(sampling_rate * 0.050))

            flag_len = len(flags)
            feat_cols = feature_vector.shape[1]
            min_sm = min(feat_cols, flag_len)
            feature_vector = feature_vector[:, 0:min_sm]
            flags = flags[0:min_sm]

            flags_new = []
            # append features and labels
            for j, fl in enumerate(flags):
                flags_new.append(classes_all.index(class_names[flags[j]]))

            flags_all = np.append(flags_all, np.array(flags_new))

            if i == 0:
                f_all = feature_vector
            else:
                f_all = np.concatenate((f_all, feature_vector), axis=1)

    # compute HMM statistics
    class_priors, transmutation_matrix, means, cov = \
        train_hmm_compute_statistics(f_all, flags_all)
    # train the HMM
    hmm = hmmlearn.hmm.GaussianHMM(class_priors.shape[0], "diag")
    hmm.covars_ = cov
    hmm.means_ = means
    hmm.startprob_ = class_priors
    hmm.transmat_ = transmutation_matrix

    # save HMM model
    with open(hmm_model_name, "wb") as f_handle:
        cpickle.dump(hmm, f_handle, protocol=cpickle.HIGHEST_PROTOCOL)
        cpickle.dump(classes_all, f_handle, protocol=cpickle.HIGHEST_PROTOCOL)
        cpickle.dump(mt_win, f_handle, protocol=cpickle.HIGHEST_PROTOCOL)
        cpickle.dump(mt_step, f_handle, protocol=cpickle.HIGHEST_PROTOCOL)

    return hmm, classes_all


def hmm_segmentation(audio_file, hmm_model_name, plot_res=False, gt_file=""):
    sampling_rate, signal = audioBasicIO.read_audio_file(audio_file)

    with open(hmm_model_name, "rb") as f_handle:
        hmm = cpickle.load(f_handle)
        classes = cpickle.load(f_handle)
        mt_win = cpickle.load(f_handle)
        mt_step = cpickle.load(f_handle)

    features, _, _ = \
        mtf.mid_feature_extraction(signal, sampling_rate,
                                   mt_win * sampling_rate,
                                   mt_step * sampling_rate,
                                   round(sampling_rate * 0.050),
                                   round(sampling_rate * 0.050))
    cm = -1
    labels_gt = np.array([])
    # apply model
    predictions = hmm.predict(features.T)
    if os.path.isfile(gt_file):
        labels_gt, classes_gt = load_ground_truth(gt_file, mt_step)
        classes = classes_gt
        cm = np.zeros((len(classes), len(classes)))
        for i in range(min(predictions.shape[0], labels_gt.shape[0])):
            cm[int(labels_gt[i]), int(predictions[i])] += 1

    acc = plot_segmentation_results(predictions, labels_gt, classes,
                                    mt_step, not plot_res)
    if acc >= 0:
        print("Overall Accuracy: {0:.2f}".format(acc))
    return predictions, classes, acc, cm


def load_ground_truth(gt_file, mt_step):
    seg_start, seg_end, seg_labels = read_segmentation_gt(gt_file)
    labels_gt, class_names_gt = segments_to_labels(seg_start, seg_end,
                                                   seg_labels, mt_step)
    labels_temp = []
    for index, label in enumerate(labels_gt):
        # "align" labels with GT
        if class_names_gt[labels_gt[index]] in class_names_gt:
            labels_temp.append(class_names_gt.index(class_names_gt[
                                                     labels_gt[index]]))
        else:
            labels_temp.append(-1)
    labels_gt = np.array(labels_temp)
    return labels_gt, class_names_gt


def mid_term_file_classification(input_file, model_name, model_type,
                                 plot_results=False, gt_file=""):
    """
    This function performs mid-term classification of an audio stream.
    Towards this end, supervised knowledge is used,
    i.e. a pre-trained classifier.
    ARGUMENTS:
        - input_file:        path of the input WAV file
        - model_name:        name of the classification model
        - model_type:        svm or knn depending on the classifier type
        - plot_results:      True if results are to be plotted using
                             matplotlib along with a set of statistics

    RETURNS:
          - segs:           a sequence of segment's endpoints: segs[i] is the
                            endpoint of the i-th segment (in seconds)
          - classes:        a sequence of class flags: class[i] is the
                            class ID of the i-th segment
    """

    if os.path.isfile(model_name):
        # Load classifier:
        if model_type == "knn":
            classifier, mean, std, class_names, mt_win, mt_step, st_win, \
            st_step, compute_beat = at.load_model_knn(model_name)
        else:
            classifier, mean, std, class_names, mt_win, mt_step, st_win, \
            st_step, compute_beat = at.load_model(model_name)

        if compute_beat:
            print("Model " + model_name + " contains long-term music features "
                                          "(beat etc) and cannot be used in "
                                          "segmentation")
            return -1, -1, -1, -1
        # load input file
        sampling_rate, signal = audioBasicIO.read_audio_file(input_file)

        # could not read file
        if sampling_rate == 0:
            return -1, -1, -1, -1

        # convert stereo (if) to mono
        signal = audioBasicIO.stereo_to_mono(signal)

        # mid-term feature extraction:
        mt_feats, _, _ = \
            mtf.mid_feature_extraction(signal, sampling_rate,
                                       mt_win * sampling_rate,
                                       mt_step * sampling_rate,
                                       round(sampling_rate * st_win),
                                       round(sampling_rate * st_step))
        posterior_matrix = []
        labels = []
        labels_gt = []
        # for each feature vector (i.e. for each fix-sized segment):
        for col_index in range(mt_feats.shape[1]):
            # normalize current feature v
            feature_vector = (mt_feats[:, col_index] - mean) / std

            # classify vector:
            label_predicted, posterior = \
                at.classifierWrapper(classifier, model_type, feature_vector)
            labels_gt.append(label_predicted)

            # update class label matrix
            labels.append(class_names[int(label_predicted)])

            # update probability matrix
            posterior_matrix.append(np.max(posterior))
        labels_gt = np.array(labels_gt)

        # 1-window smoothing
        for col_index in range(1, len(labels_gt) - 1):
            if labels_gt[col_index-1] == labels_gt[col_index + 1]:
                labels_gt[col_index] = labels_gt[col_index + 1]

        # convert fix-sized flags to segments and classes
        segs, classes = labels_to_segments(labels, mt_step)
        segs[-1] = len(signal) / float(sampling_rate)

        # Load grount-truth:
        if os.path.isfile(gt_file):
            labels_gt, classes = load_ground_truth(gt_file, mt_step)
            cm = np.zeros((len(classes), len(classes)))
            for col_index in range(min(labels_gt.shape[0], labels_gt.shape[0])):
                cm[int(labels_gt[col_index]), int(labels_gt[col_index])] += 1
        else:
            cm = []
            labels_gt = np.array([])
        acc = plot_segmentation_results(labels_gt, labels_gt,
                                        class_names, mt_step, not plot_results)
    else:
        print("mtFileClassificationError: input model_type not found!")
    if acc >= 0:
        print("Overall Accuracy: {0:.3f}".format(acc))
        return labels_gt, classes, acc, cm
    else:
        return labels_gt, class_names, acc, cm


def evaluateSegmentationClassificationDir(dir_name, model_name, method_name):
    accuracies = []

    # for each WAV file
    for i, f in enumerate(glob.glob(dir_name + os.sep + '*.wav')):
        wav_file = f
        print(wav_file)
        gt_file = f.replace('.wav', '.segments')  # open for annotated file

        if method_name.lower() in ["svm", "svm_rbf", "knn",
                                   "randomforest","gradientboosting",
                                   "extratrees"]:
            flags_ind, class_names, acc, cm_t = \
                mid_term_file_classification(wav_file, model_name, method_name,
                                             False, gt_file)
        else:
            flags_ind, class_names, acc, cm_t = hmm_segmentation(wav_file,
                                                                 model_name,
                                                                 False, gt_file)
        if acc > -1:
            if i==0:
                cm = np.copy(cm_t)
            else:                
                cm = cm + cm_t
            accuracies.append(acc)
            print(cm_t, class_names)
            print(cm)
            [rec, pre, f1] = compute_metrics(cm_t, class_names)

    cm = cm / np.sum(cm)
    [rec, pre, f1] = compute_metrics(cm, class_names)

    print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
    print("Average Accuracy: {0:.1f}".format(100.0*np.array(accuracies).mean()))
    print("Average recall: {0:.1f}".format(100.0*np.array(rec).mean()))
    print("Average precision: {0:.1f}".format(100.0*np.array(pre).mean()))
    print("Average f1: {0:.1f}".format(100.0*np.array(f1).mean()))
    print("Median Accuracy: {0:.1f}".format(100.0*np.median(np.array(accuracies))))
    print("Min Accuracy: {0:.1f}".format(100.0*np.array(accuracies).min()))
    print("Max Accuracy: {0:.1f}".format(100.0*np.array(accuracies).max()))


def silenceRemoval(x, fs, st_win, st_step, smoothWindow=0.5, weight=0.5, plot=False):
    """
    Event Detection (silence removal)
    ARGUMENTS:
         - x:                the input audio signal
         - fs:               sampling freq
         - st_win, st_step:    window size and step in seconds
         - smoothWindow:     (optinal) smooth window (in seconds)
         - weight:           (optinal) weight factor (0 < weight < 1)
                              the higher, the more strict
         - plot:             (optinal) True if results are to be plotted
    RETURNS:
         - seg_limits:    list of segment limits in seconds (e.g [[0.1, 0.9],
                          [1.4, 3.0]] means that
                          the resulting segments are (0.1 - 0.9) seconds
                          and (1.4, 3.0) seconds
    """

    if weight >= 1:
        weight = 0.99
    if weight <= 0:
        weight = 0.01

    # Step 1: feature extraction
    x = audioBasicIO.stereo_to_mono(x)
    st_feats, _ = stf.feature_extraction(x, fs, st_win * fs, st_step * fs)

    # Step 2: train binary svm classifier of low vs high energy frames
    # keep only the energy short-term sequence (2nd feature)
    st_energy = st_feats[1, :]
    en = np.sort(st_energy)
    # number of 10% of the total short-term windows
    l1 = int(len(en) / 10)
    # compute "lower" 10% energy threshold
    t1 = np.mean(en[0:l1]) + 0.000000000000001
    # compute "higher" 10% energy threshold
    t2 = np.mean(en[-l1:-1]) + 0.000000000000001
    # get all features that correspond to low energy
    class1 = st_feats[:, np.where(st_energy <= t1)[0]]
    # get all features that correspond to high energy
    class2 = st_feats[:, np.where(st_energy >= t2)[0]]
    # form the binary classification task and ...
    faets_s = [class1.T, class2.T]
    # normalize and train the respective svm probabilistic model
    # (ONSET vs SILENCE)
    [faets_s_norm, means_s, stds_s] = at.normalizeFeatures(faets_s)
    svm = at.trainSVM(faets_s_norm, 1.0)

    # Step 3: compute onset probability based on the trained svm
    prob_on_set = []
    for i in range(st_feats.shape[1]):
        # for each frame
        cur_fv = (st_feats[:, i] - means_s) / stds_s
        # get svm probability (that it belongs to the ONSET class)
        prob_on_set.append(svm.predict_proba(cur_fv.reshape(1,-1))[0][1])
    prob_on_set = np.array(prob_on_set)
    # smooth probability:
    prob_on_set = smooth_moving_avg(prob_on_set, smoothWindow / st_step)

    # Step 4A: detect onset frame indices:
    prog_on_set_sort = np.sort(prob_on_set)
    # find probability Threshold as a weighted average
    # of top 10% and lower 10% of the values
    Nt = int(prog_on_set_sort.shape[0] / 10)
    T = (np.mean((1 - weight) * prog_on_set_sort[0:Nt]) +
         weight * np.mean(prog_on_set_sort[-Nt::]))

    max_idx = np.where(prob_on_set > T)[0]
    # get the indices of the frames that satisfy the thresholding
    i = 0
    time_clusters = []
    seg_limits = []

    # Step 4B: group frame indices to onset segments
    while i < len(max_idx):
        # for each of the detected onset indices
        cur_cluster = [max_idx[i]]
        if i == len(max_idx)-1:
            break
        while max_idx[i+1] - cur_cluster[-1] <= 2:
            cur_cluster.append(max_idx[i+1])
            i += 1
            if i == len(max_idx)-1:
                break
        i += 1
        time_clusters.append(cur_cluster)
        seg_limits.append([cur_cluster[0] * st_step,
                           cur_cluster[-1] * st_step])

    # Step 5: Post process: remove very small segments:
    min_dur = 0.2
    seg_limits_2 = []
    for s in seg_limits:
        if s[1] - s[0] > min_dur:
            seg_limits_2.append(s)
    seg_limits = seg_limits_2

    if plot:
        timeX = np.arange(0, x.shape[0] / float(fs), 1.0 / fs)

        plt.subplot(2, 1, 1)
        plt.plot(timeX, x)
        for s in seg_limits:
            plt.axvline(x=s[0], color='red')
            plt.axvline(x=s[1], color='red')
        plt.subplot(2, 1, 2)
        plt.plot(np.arange(0, prob_on_set.shape[0] * st_step, st_step), 
                 prob_on_set)
        plt.title('Signal')
        for s in seg_limits:
            plt.axvline(x=s[0], color='red')
            plt.axvline(x=s[1], color='red')
        plt.title('svm Probability')
        plt.show()

    return seg_limits


def speakerDiarization(filename, n_speakers, mt_size=2.0, mt_step=0.2, 
                       st_win=0.05, lda_dim=35, plot_res=False):
    """
    ARGUMENTS:
        - filename:        the name of the WAV file to be analyzed
        - n_speakers       the number of speakers (clusters) in
                           the recording (<=0 for unknown)
        - mt_size (opt)    mid-term window size
        - mt_step (opt)    mid-term window step
        - st_win  (opt)    short-term window size
        - lda_dim (opt     LDA dimension (0 for no LDA)
        - plot_res         (opt)   0 for not plotting the results 1 for plotting
    """
    [fs, x] = audioBasicIO.read_audio_file(filename)
    x = audioBasicIO.stereo_to_mono(x)
    duration = len(x) / fs

    [classifier_1, MEAN1, STD1, classNames1, mtWin1, mtStep1, stWin1, stStep1, computeBEAT1] = at.load_model_knn(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "knnSpeakerAll"))
    [classifier_2, MEAN2, STD2, classNames2, mtWin2, mtStep2, stWin2, stStep2, computeBEAT2] = at.load_model_knn(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "knnSpeakerFemaleMale"))

    [mt_feats, st_feats, _] = mtf.mid_feature_extraction(x, fs, mt_size * fs,
                                                         mt_step * fs,
                                                         round(fs * st_win),
                                                         round(fs*st_win * 0.5))

    MidTermFeatures2 = np.zeros((mt_feats.shape[0] + len(classNames1) +
                                    len(classNames2), mt_feats.shape[1]))

    for i in range(mt_feats.shape[1]):
        cur_f1 = (mt_feats[:, i] - MEAN1) / STD1
        cur_f2 = (mt_feats[:, i] - MEAN2) / STD2
        [res, P1] = at.classifierWrapper(classifier_1, "knn", cur_f1)
        [res, P2] = at.classifierWrapper(classifier_2, "knn", cur_f2)
        MidTermFeatures2[0:mt_feats.shape[0], i] = mt_feats[:, i]
        MidTermFeatures2[mt_feats.shape[0]:mt_feats.shape[0]+len(classNames1), i] = P1 + 0.0001
        MidTermFeatures2[mt_feats.shape[0] + len(classNames1)::, i] = P2 + 0.0001

    mt_feats = MidTermFeatures2    # TODO
    iFeaturesSelect = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 41,
                       42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]

    mt_feats = mt_feats[iFeaturesSelect, :]

    (mt_feats_norm, MEAN, STD) = at.normalizeFeatures([mt_feats.T])
    mt_feats_norm = mt_feats_norm[0].T
    n_wins = mt_feats.shape[1]

    # remove outliers:
    dist_all = np.sum(distance.squareform(distance.pdist(mt_feats_norm.T)),
                         axis=0)
    m_dist_all = np.mean(dist_all)
    i_non_outliers = np.nonzero(dist_all < 1.2 * m_dist_all)[0]

    # TODO: Combine energy threshold for outlier removal:
    #EnergyMin = np.min(mt_feats[1,:])
    #EnergyMean = np.mean(mt_feats[1,:])
    #Thres = (1.5*EnergyMin + 0.5*EnergyMean) / 2.0
    #i_non_outliers = np.nonzero(mt_feats[1,:] > Thres)[0]
    #print i_non_outliers

    perOutLier = (100.0 * (n_wins - i_non_outliers.shape[0])) / n_wins
    mt_feats_norm_or = mt_feats_norm
    mt_feats_norm = mt_feats_norm[:, i_non_outliers]

    # LDA dimensionality reduction:
    if lda_dim > 0:
        #[mt_feats_to_red, _, _] = aF.mtFeatureExtraction(x, fs, mt_size * fs,
        # st_win * fs, round(fs*st_win), round(fs*st_win));
        # extract mid-term features with minimum step:
        mt_win_ratio = int(round(mt_size / st_win))
        mt_step_ratio = int(round(st_win / st_win))
        mt_feats_to_red = []
        num_of_features = len(st_feats)
        num_of_stats = 2
        #for i in range(num_of_stats * num_of_features + 1):
        for i in range(num_of_stats * num_of_features):
            mt_feats_to_red.append([])

        for i in range(num_of_features):  # for each of the short-term features:
            curPos = 0
            N = len(st_feats[i])
            while (curPos < N):
                N1 = curPos
                N2 = curPos + mt_win_ratio
                if N2 > N:
                    N2 = N
                curStFeatures = st_feats[i][N1:N2]
                mt_feats_to_red[i].append(np.mean(curStFeatures))
                mt_feats_to_red[i+num_of_features].append(np.std(curStFeatures))
                curPos += mt_step_ratio
        mt_feats_to_red = np.array(mt_feats_to_red)
        mt_feats_to_red_2 = np.zeros((mt_feats_to_red.shape[0] +
                                        len(classNames1) + len(classNames2),
                                         mt_feats_to_red.shape[1]))
        for i in range(mt_feats_to_red.shape[1]):
            cur_f1 = (mt_feats_to_red[:, i] - MEAN1) / STD1
            cur_f2 = (mt_feats_to_red[:, i] - MEAN2) / STD2
            [res, P1] = at.classifierWrapper(classifier_1, "knn", cur_f1)
            [res, P2] = at.classifierWrapper(classifier_2, "knn", cur_f2)
            mt_feats_to_red_2[0:mt_feats_to_red.shape[0], i] = mt_feats_to_red[:, i]
            mt_feats_to_red_2[mt_feats_to_red.shape[0]:mt_feats_to_red.shape[0] + len(classNames1), i] = P1 + 0.0001
            mt_feats_to_red_2[mt_feats_to_red.shape[0]+len(classNames1)::, i] = P2 + 0.0001
        mt_feats_to_red = mt_feats_to_red_2
        mt_feats_to_red = mt_feats_to_red[iFeaturesSelect, :]
        #mt_feats_to_red += np.random.rand(mt_feats_to_red.shape[0], mt_feats_to_red.shape[1]) * 0.0000010
        (mt_feats_to_red, MEAN, STD) = at.normalizeFeatures([mt_feats_to_red.T])
        mt_feats_to_red = mt_feats_to_red[0].T
        #dist_all = np.sum(distance.squareform(distance.pdist(mt_feats_to_red.T)), axis=0)
        #m_dist_all = np.mean(dist_all)
        #iNonOutLiers2 = np.nonzero(dist_all < 3.0*m_dist_all)[0]
        #mt_feats_to_red = mt_feats_to_red[:, iNonOutLiers2]
        Labels = np.zeros((mt_feats_to_red.shape[1], ));
        LDAstep = 1.0
        LDAstepRatio = LDAstep / st_win
        #print LDAstep, LDAstepRatio
        for i in range(Labels.shape[0]):
            Labels[i] = int(i*st_win/LDAstepRatio);        
        clf = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=lda_dim)
        clf.fit(mt_feats_to_red.T, Labels)
        mt_feats_norm = (clf.transform(mt_feats_norm.T)).T

    if n_speakers <= 0:
        s_range = range(2, 10)
    else:
        s_range = [n_speakers]
    clsAll = []
    sil_all = []
    centersAll = []
    
    for iSpeakers in s_range:        
        k_means = sklearn.cluster.KMeans(n_clusters=iSpeakers)
        k_means.fit(mt_feats_norm.T)
        cls = k_means.labels_        
        means = k_means.cluster_centers_

        # Y = distance.squareform(distance.pdist(mt_feats_norm.T))
        clsAll.append(cls)
        centersAll.append(means)
        sil_1 = []; sil_2 = []
        for c in range(iSpeakers):
            # for each speaker (i.e. for each extracted cluster)
            clust_per_cent = np.nonzero(cls == c)[0].shape[0] / \
                             float(len(cls))
            if clust_per_cent < 0.020:
                sil_1.append(0.0)
                sil_2.append(0.0)
            else:
                # get subset of feature vectors
                mt_feats_norm_temp = mt_feats_norm[:, cls==c]
                # compute average distance between samples
                # that belong to the cluster (a values)
                Yt = distance.pdist(mt_feats_norm_temp.T)
                sil_1.append(np.mean(Yt)*clust_per_cent)
                silBs = []
                for c2 in range(iSpeakers):
                    # compute distances from samples of other clusters
                    if c2 != c:
                        clust_per_cent_2 = np.nonzero(cls == c2)[0].shape[0] /\
                                           float(len(cls))
                        MidTermFeaturesNormTemp2 = mt_feats_norm[:, cls == c2]
                        Yt = distance.cdist(mt_feats_norm_temp.T, 
                                            MidTermFeaturesNormTemp2.T)
                        silBs.append(np.mean(Yt)*(clust_per_cent
                                                     + clust_per_cent_2)/2.0)
                silBs = np.array(silBs)
                # ... and keep the minimum value (i.e.
                # the distance from the "nearest" cluster)
                sil_2.append(min(silBs))
        sil_1 = np.array(sil_1); 
        sil_2 = np.array(sil_2); 
        sil = []
        for c in range(iSpeakers):
            # for each cluster (speaker) compute silhouette
            sil.append( ( sil_2[c] - sil_1[c]) / (max(sil_2[c],
                                                      sil_1[c]) + 0.00001))
        # keep the AVERAGE SILLOUETTE
        sil_all.append(np.mean(sil))

    imax = np.argmax(sil_all)
    # optimal number of clusters
    nSpeakersFinal = s_range[imax]

    # generate the final set of cluster labels
    # (important: need to retrieve the outlier windows:
    # this is achieved by giving them the value of their
    # nearest non-outlier window)
    cls = np.zeros((n_wins,))
    for i in range(n_wins):
        j = np.argmin(np.abs(i-i_non_outliers))        
        cls[i] = clsAll[imax][j]
        
    # Post-process method 1: hmm smoothing
    for i in range(1):
        # hmm training
        start_prob, transmat, means, cov = \
            train_hmm_compute_statistics(mt_feats_norm_or, cls)
        hmm = hmmlearn.hmm.GaussianHMM(start_prob.shape[0], "diag")
        hmm.startprob_ = start_prob
        hmm.transmat_ = transmat            
        hmm.means_ = means; hmm.covars_ = cov
        cls = hmm.predict(mt_feats_norm_or.T)                    
    
    # Post-process method 2: median filtering:
    cls = scipy.signal.medfilt(cls, 13)
    cls = scipy.signal.medfilt(cls, 11)

    sil = sil_all[imax]
    class_names = ["speaker{0:d}".format(c) for c in range(nSpeakersFinal)];


    # load ground-truth if available
    gt_file = filename.replace('.wav', '.segments')
    # if groundturh exists
    if os.path.isfile(gt_file):
        [seg_start, seg_end, seg_labs] = read_segmentation_gt(gt_file)
        flags_gt, class_names_gt = segments_to_labels(seg_start, seg_end, seg_labs, mt_step)

    if plot_res:
        fig = plt.figure()    
        if n_speakers > 0:
            ax1 = fig.add_subplot(111)
        else:
            ax1 = fig.add_subplot(211)
        ax1.set_yticks(np.array(range(len(class_names))))
        ax1.axis((0, duration, -1, len(class_names)))
        ax1.set_yticklabels(class_names)
        ax1.plot(np.array(range(len(cls)))*mt_step+mt_step/2.0, cls)

    if os.path.isfile(gt_file):
        if plot_res:
            ax1.plot(np.array(range(len(flags_gt))) *
                     mt_step + mt_step / 2.0, flags_gt, 'r')
        purity_cluster_m, purity_speaker_m = \
            evaluate_speaker_diarization(cls, flags_gt)
        print("{0:.1f}\t{1:.1f}".format(100 * purity_cluster_m,
                                        100 * purity_speaker_m))
        if plot_res:
            plt.title("Cluster purity: {0:.1f}% - "
                      "Speaker purity: {1:.1f}%".format(100 * purity_cluster_m,
                                                        100 * purity_speaker_m))
    if plot_res:
        plt.xlabel("time (seconds)")
        #print s_range, sil_all    
        if n_speakers<=0:
            plt.subplot(212)
            plt.plot(s_range, sil_all)
            plt.xlabel("number of clusters");
            plt.ylabel("average clustering's sillouette");
        plt.show()
    return cls


def speakerDiarizationEvaluateScript(folder_name, ldas):
    """
        This function prints the cluster purity and speaker purity for
        each WAV file stored in a provided directory (.SEGMENT files
         are needed as ground-truth)
        ARGUMENTS:
            - folder_name:     the full path of the folder where the WAV and
                              SEGMENT (ground-truth) files are stored
            - ldas:           a list of LDA dimensions (0 for no LDA)
    """
    types = ('*.wav',  )
    wavFilesList = []
    for files in types:
        wavFilesList.extend(glob.glob(os.path.join(folder_name, files)))    
    
    wavFilesList = sorted(wavFilesList)

    # get number of unique speakers per file (from ground-truth)    
    N = []
    for wav_file in wavFilesList:        
        gt_file = wav_file.replace('.wav', '.segments');
        if os.path.isfile(gt_file):
            [seg_start, seg_end, seg_labs] = read_segmentation_gt(gt_file)
            N.append(len(list(set(seg_labs))))
        else:
            N.append(-1)
    
    for l in ldas:
        print("LDA = {0:d}".format(l))
        for i, wav_file in enumerate(wavFilesList):
            speakerDiarization(wav_file, N[i], 2.0, 0.2, 0.05, l, plot_res=False)
        print
        
def musicThumbnailing(x, fs, short_term_size=1.0, short_term_step=0.5, 
                      thumb_size=10.0, limit_1 = 0, limit_2 = 1):
    """
    This function detects instances of the most representative part of a
    music recording, also called "music thumbnails".
    A technique similar to the one proposed in [1], however a wider set of
    audio features is used instead of chroma features.
    In particular the following steps are followed:
     - Extract short-term audio features. Typical short-term window size: 1 second
     - Compute the self-similarity matrix, i.e. all pairwise similarities
       between feature vectors
     - Apply a diagonal mask is as a moving average filter on the values of the
       self-similarty matrix.
       The size of the mask is equal to the desirable thumbnail length.
     - Find the position of the maximum value of the new (filtered)
       self-similarity matrix. The audio segments that correspond to the
       diagonial around that position are the selected thumbnails
    

    ARGUMENTS:
     - x:            input signal
     - fs:            sampling frequency
     - short_term_size:     window size (in seconds)
     - short_term_step:    window step (in seconds)
     - thumb_size:    desider thumbnail size (in seconds)
    
    RETURNS:
     - A1:            beginning of 1st thumbnail (in seconds)
     - A2:            ending of 1st thumbnail (in seconds)
     - B1:            beginning of 2nd thumbnail (in seconds)
     - B2:            ending of 2nd thumbnail (in seconds)

    USAGE EXAMPLE:
       import audioFeatureExtraction as aF
     [fs, x] = basicIO.readAudioFile(input_file)
     [A1, A2, B1, B2] = musicThumbnailing(x, fs)

    [1] Bartsch, M. A., & Wakefield, G. H. (2005). Audio thumbnailing
    of popular music using chroma-based representations.
    Multimedia, IEEE Transactions on, 7(1), 96-104.
    """
    x = audioBasicIO.stereo_to_mono(x);
    # feature extraction:
    st_feats, _ = stf.feature_extraction(x, fs, fs * short_term_size,
                                         fs * short_term_step)

    # self-similarity matrix
    S = self_similarity_matrix(st_feats)

    # moving filter:
    M = int(round(thumb_size / short_term_step))
    B = np.eye(M,M)
    S = scipy.signal.convolve2d(S, B, 'valid')


    # post-processing (remove main diagonal elements)
    min_sm = np.min(S)
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if abs(i-j) < 5.0 / short_term_step or i > j:
                S[i,j] = min_sm;

    # find max position:
    S[0:int(limit_1 * S.shape[0]), :] = min_sm
    S[:, 0:int(limit_1 * S.shape[0])] = min_sm
    S[int(limit_2 * S.shape[0])::, :] = min_sm
    S[:, int(limit_2 * S.shape[0])::] = min_sm

    maxVal = np.max(S)        
    [I, J] = np.unravel_index(S.argmax(), S.shape)
    #plt.imshow(S)
    #plt.show()
    # expand:
    i1 = I
    i2 = I
    j1 = J
    j2 = J

    while i2-i1<M: 
        if i1 <=0 or j1<=0 or i2 >= S.shape[0]-2 or j2 >= S.shape[1]-2:
            break
        if S[i1-1, j1-1] > S[i2 + 1, j2 + 1]:
            i1 -= 1
            j1 -= 1            
        else:            
            i2 += 1
            j2 += 1            

    return short_term_step * i1, short_term_step * i2, \
           short_term_step * j1, short_term_step * j2, S


