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
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../"))
import pyAudioAnalysis.audioBasicIO as audioBasicIO
import pyAudioAnalysis.audioTrainTest as at
import pyAudioAnalysis.MidTermFeatures as mtf
import pyAudioAnalysis.ShortTermFeatures as stf


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
    scaler = StandardScaler()
    norm_feature_vectors = scaler.fit_transform(feature_vectors.T).T
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

    if len(labels)==1:
        segs = [0, window]
        classes = labels
        return segs, classes


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
        reader = csv.reader(f_handle, delimiter='\t')
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


def train_hmm_from_file(wav_file, gt_file, hmm_model_name, mid_window, mid_step):
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
    flags, class_names = segments_to_labels(seg_start, seg_end, seg_labs, mid_step)
    sampling_rate, signal = audioBasicIO.read_audio_file(wav_file)
    features, _, _ = \
        mtf.mid_feature_extraction(signal, sampling_rate,
                                   mid_window * sampling_rate,
                                   mid_step * sampling_rate,
                                   round(sampling_rate * 0.050),
                                   round(sampling_rate * 0.050))
    class_priors, transumation_matrix, means, cov = \
        train_hmm_compute_statistics(features, flags)
    hmm = hmmlearn.hmm.GaussianHMM(class_priors.shape[0], "diag")

    hmm.covars_ = cov
    hmm.means_ = means
    hmm.startprob_ = class_priors
    hmm.transmat_ = transumation_matrix

    save_hmm(hmm_model_name, hmm, class_names, mid_window, mid_step)

    return hmm, class_names


def train_hmm_from_directory(folder_path, hmm_model_name, mid_window, mid_step):
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
    class_names_all = []
    for i, f in enumerate(glob.glob(folder_path + os.sep + '*.wav')):
        # for each WAV file
        wav_file = f
        gt_file = f.replace('.wav', '.segments')
        if os.path.isfile(gt_file):
            seg_start, seg_end, seg_labs = read_segmentation_gt(gt_file)
            flags, class_names = \
                segments_to_labels(seg_start, seg_end, seg_labs, mid_step)
            for c in class_names:
                # update class names:
                if c not in class_names_all:
                    class_names_all.append(c)
            sampling_rate, signal = audioBasicIO.read_audio_file(wav_file)
            feature_vector, _, _ = \
                mtf.mid_feature_extraction(signal, sampling_rate,
                                           mid_window * sampling_rate,
                                           mid_step * sampling_rate,
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
                flags_new.append(class_names_all.index(class_names_all[flags[j]]))

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

    save_hmm(hmm_model_name, hmm, class_names_all, mid_window, mid_step)

    return hmm, class_names_all


def save_hmm(hmm_model_name, model, classes, mid_window, mid_step):
    """Save HMM model"""
    with open(hmm_model_name, "wb") as f_handle:
        cpickle.dump(model, f_handle, protocol=cpickle.HIGHEST_PROTOCOL)
        cpickle.dump(classes, f_handle, protocol=cpickle.HIGHEST_PROTOCOL)
        cpickle.dump(mid_window, f_handle, protocol=cpickle.HIGHEST_PROTOCOL)
        cpickle.dump(mid_step, f_handle, protocol=cpickle.HIGHEST_PROTOCOL)


def hmm_segmentation(audio_file, hmm_model_name, plot_results=False,
                     gt_file=""):
    sampling_rate, signal = audioBasicIO.read_audio_file(audio_file)

    with open(hmm_model_name, "rb") as f_handle:
        hmm = cpickle.load(f_handle)
        class_names = cpickle.load(f_handle)
        mid_window = cpickle.load(f_handle)
        mid_step = cpickle.load(f_handle)

    features, _, _ = \
        mtf.mid_feature_extraction(signal, sampling_rate,
                                   mid_window * sampling_rate,
                                   mid_step * sampling_rate,
                                   round(sampling_rate * 0.050),
                                   round(sampling_rate * 0.050))

    # apply model
    labels = hmm.predict(features.T)
    labels_gt, class_names_gt, accuracy, cm = \
        load_ground_truth(gt_file, labels, class_names, mid_step, plot_results)
    return labels, class_names, accuracy, cm


def load_ground_truth_segments(gt_file, mt_step):
    seg_start, seg_end, seg_labels = read_segmentation_gt(gt_file)
    labels, class_names = segments_to_labels(seg_start, seg_end, seg_labels,
                                             mt_step)
    labels_temp = []
    for index, label in enumerate(labels):
        # "align" labels with GT
        if class_names[labels[index]] in class_names:
            labels_temp.append(class_names.index(class_names[
                                                     labels[index]]))
        else:
            labels_temp.append(-1)
    labels = np.array(labels_temp)
    return labels, class_names


def calculate_confusion_matrix(predictions, ground_truth, classes):
    cm = np.zeros((len(classes), len(classes)))
    for index in range(min(predictions.shape[0], ground_truth.shape[0])):
        cm[int(ground_truth[index]), int(predictions[index])] += 1
    return cm


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
        - gt_file:           path to the ground truth file, if exists, 
                             for calculating classification performance
    RETURNS:
    labels, class_names, accuracy, cm
          - labels:         a sequence of segment's labels: segs[i] is the label
                            of the i-th segment
          - class_names:    a string sequence of class_names used in classification:
                            class_names[i] is the name of classes[i]
          - accuracy:       the accuracy of the classification.
          - cm:             the confusion matrix of this classification
    """
    labels = []
    accuracy = 0.0
    class_names = []
    cm = np.array([])
    if not os.path.isfile(model_name):
        print("mtFileClassificationError: input model_type not found!")
        return labels, class_names, accuracy, cm

    # Load classifier:
    if model_type == "knn":
        classifier, mean, std, class_names, mt_win, mid_step, st_win, \
         st_step, compute_beat = at.load_model_knn(model_name)
    else:
        classifier, mean, std, class_names, mt_win, mid_step, st_win, \
         st_step, compute_beat = at.load_model(model_name)
    if compute_beat:
        print("Model " + model_name + " contains long-term music features "
                                      "(beat etc) and cannot be used in "
                                      "segmentation")
        return labels, class_names, accuracy, cm
    # load input file
    sampling_rate, signal = audioBasicIO.read_audio_file(input_file)

    # could not read file
    if sampling_rate == 0:
        return labels, class_names, accuracy, cm

    # convert stereo (if) to mono
    signal = audioBasicIO.stereo_to_mono(signal)

    # mid-term feature extraction:
    mt_feats, _, _ = \
        mtf.mid_feature_extraction(signal, sampling_rate,
                                   mt_win * sampling_rate,
                                   mid_step * sampling_rate,
                                   round(sampling_rate * st_win),
                                   round(sampling_rate * st_step))
    posterior_matrix = []

    # for each feature vector (i.e. for each fix-sized segment):
    for col_index in range(mt_feats.shape[1]):
        # normalize current feature v
        feature_vector = (mt_feats[:, col_index] - mean) / std

        # classify vector:
        label_predicted, posterior = \
            at.classifier_wrapper(classifier, model_type, feature_vector)
        labels.append(label_predicted)

        # update probability matrix
        posterior_matrix.append(np.max(posterior))
    labels = np.array(labels)

    # convert fix-sized flags to segments and classes
    segs, classes = labels_to_segments(labels, mid_step)
    for i in range(len(segs)):
        print(segs[i], classes[i])
    segs[-1] = len(signal) / float(sampling_rate)
    # Load grount-truth:
    labels_gt, class_names_gt, accuracy, cm = \
        load_ground_truth(gt_file, labels, class_names, mid_step, plot_results)

    return labels, class_names, accuracy, cm


def load_ground_truth(gt_file, labels, class_names, mid_step, plot_results):
    accuracy = 0
    cm = np.array([])
    labels_gt = np.array([])
    if os.path.isfile(gt_file):
        # load ground truth and class names
        labels_gt, class_names_gt = load_ground_truth_segments(gt_file,
                                                               mid_step)

        # map predicted labels to ground truth class names
        # Note: if a predicted label does not belong to the ground truth
        #       classes --> -1
        labels_new = []
        for il, l in enumerate(labels):
            if class_names[int(l)] in class_names_gt:
                labels_new.append(class_names_gt.index(class_names[int(l)]))
            else:
                labels_new.append(-1)
        labels_new = np.array(labels_new)
        cm = calculate_confusion_matrix(labels_new, labels_gt, class_names_gt)

        accuracy = plot_segmentation_results(labels_new, labels_gt,
                                             class_names_gt, mid_step, 
                                             not plot_results)
        if accuracy >= 0:
            print("Overall Accuracy: {0:.2f}".format(accuracy))

    return labels_gt, class_names, accuracy, cm


def evaluate_segmentation_classification_dir(dir_name, model_name, method_name):

    accuracies = []
    class_names = []
    cm_total = np.array([])
    for index, wav_file in enumerate(glob.glob(dir_name + os.sep + '*.wav')):
        print(wav_file)

        gt_file = wav_file.replace('.wav', '.segments')

        if method_name.lower() in ["svm", "svm_rbf", "knn", "randomforest",
                                   "gradientboosting", "extratrees"]:
            flags_ind, class_names, accuracy, cm_temp = \
                mid_term_file_classification(wav_file, model_name, method_name,
                                             False, gt_file)
        else:
            flags_ind, class_names, accuracy, cm_temp = \
                hmm_segmentation(wav_file, model_name, False, gt_file)
        if accuracy > 0:
            if not index:
                cm_total = np.copy(cm_temp)
            else:
                cm_total = cm_total + cm_temp
            accuracies.append(accuracy)
            print(cm_temp, class_names)
            print(cm_total)

    if len(cm_total.shape) > 1:
        cm_total = cm_total / np.sum(cm_total)
        rec, pre, f1 = compute_metrics(cm_total, class_names)

        print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
        print("Average Accuracy: {0:.1f}".
              format(100.0*np.array(accuracies).mean()))
        print("Average recall: {0:.1f}".format(100.0*np.array(rec).mean()))
        print("Average precision: {0:.1f}".format(100.0*np.array(pre).mean()))
        print("Average f1: {0:.1f}".format(100.0*np.array(f1).mean()))
        print("Median Accuracy: {0:.1f}".
              format(100.0*np.median(np.array(accuracies))))
        print("Min Accuracy: {0:.1f}".format(100.0*np.array(accuracies).min()))
        print("Max Accuracy: {0:.1f}".format(100.0*np.array(accuracies).max()))
    else:
        print("Confusion matrix was empty, accuracy for every file was 0")


def silence_removal(signal, sampling_rate, st_win, st_step, smooth_window=0.5,
                    weight=0.5, plot=False):
    """
    Event Detection (silence removal)
    ARGUMENTS:
         - signal:                the input audio signal
         - sampling_rate:               sampling freq
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
    signal = audioBasicIO.stereo_to_mono(signal)
    st_feats, _ = stf.feature_extraction(signal, sampling_rate,
                                         st_win * sampling_rate,
                                         st_step * sampling_rate)

    # Step 2: train binary svm classifier of low vs high energy frames
    # keep only the energy short-term sequence (2nd feature)
    st_energy = st_feats[1, :]
    en = np.sort(st_energy)
    # number of 10% of the total short-term windows
    st_windows_fraction = int(len(en) / 10)

    # compute "lower" 10% energy threshold
    low_threshold = np.mean(en[0:st_windows_fraction]) + 1e-15

    # compute "higher" 10% energy threshold
    high_threshold = np.mean(en[-st_windows_fraction:-1]) + 1e-15

    # get all features that correspond to low energy
    low_energy = st_feats[:, np.where(st_energy <= low_threshold)[0]]

    # get all features that correspond to high energy
    high_energy = st_feats[:, np.where(st_energy >= high_threshold)[0]]

    # form the binary classification task and ...
    features = [low_energy.T, high_energy.T]
    # normalize and train the respective svm probabilistic model

    # (ONSET vs SILENCE)
    features, labels = at.features_to_matrix(features)
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)
    mean = scaler.mean_
    std = scaler.scale_
    svm = at.train_svm(features_norm, labels, 1.0)

    # Step 3: compute onset probability based on the trained svm
    prob_on_set = []
    for index in range(st_feats.shape[1]):
        # for each frame
        cur_fv = (st_feats[:, index] - mean) / std
        # get svm probability (that it belongs to the ONSET class)
        prob_on_set.append(svm.predict_proba(cur_fv.reshape(1, -1))[0][1])
    prob_on_set = np.array(prob_on_set)

    # smooth probability:
    prob_on_set = smooth_moving_avg(prob_on_set, smooth_window / st_step)

    # Step 4A: detect onset frame indices:
    prog_on_set_sort = np.sort(prob_on_set)

    # find probability Threshold as a weighted average
    # of top 10% and lower 10% of the values
    nt = int(prog_on_set_sort.shape[0] / 10)
    threshold = (np.mean((1 - weight) * prog_on_set_sort[0:nt]) +
         weight * np.mean(prog_on_set_sort[-nt::]))

    max_indices = np.where(prob_on_set > threshold)[0]
    # get the indices of the frames that satisfy the thresholding
    index = 0
    seg_limits = []
    time_clusters = []

    # Step 4B: group frame indices to onset segments
    while index < len(max_indices):
        # for each of the detected onset indices
        cur_cluster = [max_indices[index]]
        if index == len(max_indices)-1:
            break
        while max_indices[index+1] - cur_cluster[-1] <= 2:
            cur_cluster.append(max_indices[index+1])
            index += 1
            if index == len(max_indices)-1:
                break
        index += 1
        time_clusters.append(cur_cluster)
        seg_limits.append([cur_cluster[0] * st_step,
                           cur_cluster[-1] * st_step])

    # Step 5: Post process: remove very small segments:
    min_duration = 0.2
    seg_limits_2 = []
    for s_lim in seg_limits:
        if s_lim[1] - s_lim[0] > min_duration:
            seg_limits_2.append(s_lim)
    seg_limits = seg_limits_2

    if plot:
        time_x = np.arange(0, signal.shape[0] / float(sampling_rate), 1.0 /
                           sampling_rate)

        plt.subplot(2, 1, 1)
        plt.plot(time_x, signal)
        for s_lim in seg_limits:
            plt.axvline(x=s_lim[0], color='red')
            plt.axvline(x=s_lim[1], color='red')
        plt.subplot(2, 1, 2)
        plt.plot(np.arange(0, prob_on_set.shape[0] * st_step, st_step), 
                 prob_on_set)
        plt.title('Signal')
        for s_lim in seg_limits:
            plt.axvline(x=s_lim[0], color='red')
            plt.axvline(x=s_lim[1], color='red')
        plt.title('svm Probability')
        plt.show()

    return seg_limits


def speaker_diarization(filename, n_speakers, mid_window=1.0, mid_step=0.1,
                        short_window=0.1, lda_dim=0, plot_res=False):
    """
    ARGUMENTS:
        - filename:        the name of the WAV file to be analyzed
        - n_speakers       the number of speakers (clusters) in
                           the recording (<=0 for unknown)
        - mid_window (opt)    mid-term window size
        - mid_step (opt)    mid-term window step
        - short_window  (opt)    short-term window size
        - lda_dim (opt     LDA dimension (0 for no LDA)
        - plot_res         (opt)   0 for not plotting the results 1 for plotting
    """
    sampling_rate, signal = audioBasicIO.read_audio_file(filename)
    signal = audioBasicIO.stereo_to_mono(signal)
    duration = len(signal) / sampling_rate

    base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "data/models")

    classifier_all, mean_all, std_all, class_names_all, _, _, _, _, _ = \
        at.load_model(os.path.join(base_dir, "svm_rbf_speaker_10"))
    classifier_fm, mean_fm, std_fm, class_names_fm, _, _, _, _,  _ = \
        at.load_model(os.path.join(base_dir, "svm_rbf_speaker_male_female"))


    mid_feats, st_feats, a = \
        mtf.mid_feature_extraction(signal, sampling_rate,
                                   mid_window * sampling_rate,
                                   mid_step * sampling_rate,
                                   round(sampling_rate * 0.05),
                                   round(sampling_rate * 0.05))

    mid_term_features = np.zeros((mid_feats.shape[0] + len(class_names_all) +
                                  len(class_names_fm), mid_feats.shape[1]))
    for index in range(mid_feats.shape[1]):
        feature_norm_all = (mid_feats[:, index] - mean_all) / std_all
        feature_norm_fm = (mid_feats[:, index] - mean_fm) / std_fm
        _, p1 = at.classifier_wrapper(classifier_all, "svm_rbf", feature_norm_all)
        _, p2 = at.classifier_wrapper(classifier_fm, "svm_rbf", feature_norm_fm)
        start = mid_feats.shape[0]
        end = mid_feats.shape[0] + len(class_names_all)
        mid_term_features[0:mid_feats.shape[0], index] = mid_feats[:, index]
        mid_term_features[start:end, index] = p1 + 1e-4
        mid_term_features[end::, index] = p2 + 1e-4
    # normalize features:
    scaler = StandardScaler()
    mid_feats_norm = scaler.fit_transform(mid_term_features.T)

    # remove outliers:
    dist_all = np.sum(distance.squareform(distance.pdist(mid_feats_norm.T)),
                      axis=0)
    m_dist_all = np.mean(dist_all)
    i_non_outliers = np.nonzero(dist_all < 1.1 * m_dist_all)[0]

    # TODO: Combine energy threshold for outlier removal:
    # EnergyMin = np.min(mt_feats[1,:])
    # EnergyMean = np.mean(mt_feats[1,:])
    # Thres = (1.5*EnergyMin + 0.5*EnergyMean) / 2.0
    # i_non_outliers = np.nonzero(mt_feats[1,:] > Thres)[0]
    # print i_non_outliers

    mt_feats_norm_or = mid_feats_norm
    mid_feats_norm = mid_feats_norm[:, i_non_outliers]

    # LDA dimensionality reduction:
    if lda_dim > 0:

        # extract mid-term features with minimum step:
        window_ratio = int(round(mid_window / short_window))
        step_ratio = int(round(short_window / short_window))
        mt_feats_to_red = []
        num_of_features = len(st_feats)
        num_of_stats = 2
        for index in range(num_of_stats * num_of_features):
            mt_feats_to_red.append([])

        # for each of the short-term features:
        for index in range(num_of_features):
            cur_pos = 0
            feat_len = len(st_feats[index])
            while cur_pos < feat_len:
                n1 = cur_pos
                n2 = cur_pos + window_ratio
                if n2 > feat_len:
                    n2 = feat_len
                short_features = st_feats[index][n1:n2]
                mt_feats_to_red[index].append(np.mean(short_features))
                mt_feats_to_red[index + num_of_features].\
                    append(np.std(short_features))
                cur_pos += step_ratio
        mt_feats_to_red = np.array(mt_feats_to_red)
        mt_feats_to_red_2 = np.zeros((mt_feats_to_red.shape[0] +
                                      len(class_names_all) +
                                      len(class_names_fm),
                                      mt_feats_to_red.shape[1]))
        limit = mt_feats_to_red.shape[0] + len(class_names_all)
        for index in range(mt_feats_to_red.shape[1]):
            feature_norm_all = (mt_feats_to_red[:, index] - mean_all) / std_all
            feature_norm_fm = (mt_feats_to_red[:, index] - mean_fm) / std_fm
            _, p1 = at.classifier_wrapper(classifier_all, "svm_rbf",
                                          feature_norm_all)
            _, p2 = at.classifier_wrapper(classifier_fm, "svm_rbf", feature_norm_fm)
            mt_feats_to_red_2[0:mt_feats_to_red.shape[0], index] = \
                mt_feats_to_red[:, index]
            mt_feats_to_red_2[mt_feats_to_red.shape[0]:limit, index] = p1 + 1e-4
            mt_feats_to_red_2[limit::, index] = p2 + 1e-4
        mt_feats_to_red = mt_feats_to_red_2
        scaler = StandardScaler()
        mt_feats_to_red = scaler.fit_transform(mt_feats_to_red.T).T
        labels = np.zeros((mt_feats_to_red.shape[1], ))
        lda_step = 1.0
        lda_step_ratio = lda_step / short_window
        for index in range(labels.shape[0]):
            labels[index] = int(index * short_window / lda_step_ratio)
        clf = sklearn.discriminant_analysis.\
            LinearDiscriminantAnalysis(n_components=lda_dim)
        mid_feats_norm = clf.fit_transform(mt_feats_to_red.T, labels)
        #clf.fit(mt_feats_to_red.T, labels)
        #mid_feats_norm = (clf.transform(mid_feats_norm.T)).T
    if n_speakers <= 0:
        s_range = range(2, 10)
    else:
        s_range = [n_speakers]
    cluster_labels = []
    sil_all = []
    cluster_centers = []
    
    for speakers in s_range:
        k_means = sklearn.cluster.KMeans(n_clusters=speakers)
        k_means.fit(mid_feats_norm)
        cls = k_means.labels_ 
        cluster_labels.append(cls)
#        cluster_centers.append(means)
        sil_1 = []; sil_2 = []
        for c in range(speakers):
            # for each speaker (i.e. for each extracted cluster)
            clust_per_cent = np.nonzero(cls == c)[0].shape[0] / float(len(cls))
            if clust_per_cent < 0.020:
                sil_1.append(0.0)
                sil_2.append(0.0)
            else:
                # get subset of feature vectors
                mt_feats_norm_temp = mid_feats_norm[cls == c, :]
                # compute average distance between samples
                # that belong to the cluster (a values)
                dist = distance.pdist(mt_feats_norm_temp.T)
                sil_1.append(np.mean(dist)*clust_per_cent)
                sil_temp = []
                for c2 in range(speakers):
                    # compute distances from samples of other clusters
                    if c2 != c:
                        clust_per_cent_2 = np.nonzero(cls == c2)[0].shape[0] /\
                                           float(len(cls))
                        mid_features_temp = mid_feats_norm[cls == c2, :]
                        dist = distance.cdist(mt_feats_norm_temp,
                                              mid_features_temp)
                        sil_temp.append(np.mean(dist)*(clust_per_cent
                                                       + clust_per_cent_2)/2.0)
                sil_temp = np.array(sil_temp)
                # ... and keep the minimum value (i.e.
                # the distance from the "nearest" cluster)
                sil_2.append(min(sil_temp))
        sil_1 = np.array(sil_1)
        sil_2 = np.array(sil_2)
        sil = []
        for c in range(speakers):
            # for each cluster (speaker) compute silhouette
            sil.append((sil_2[c] - sil_1[c]) / (max(sil_2[c], sil_1[c]) + 1e-5))
        # keep the AVERAGE SILLOUETTE
        sil_all.append(np.mean(sil))

    imax = int(np.argmax(sil_all))
    # optimal number of clusters
    num_speakers = s_range[imax]

    # generate the final set of cluster labels
    # (important: need to retrieve the outlier windows:
    # this is achieved by giving them the value of their
    # nearest non-outlier window)
#    print(cls)
#    cls = np.zeros((n_wins,))
#    for index in range(n_wins):
#        j = np.argmin(np.abs(index-i_non_outliers))
#        cls[index] = cluster_labels[imax][j]
    # Post-process method 1: hmm smoothing
    if lda_dim <= 0 :
        for index in range(1):
            # hmm training
            start_prob, transmat, means, cov = \
                train_hmm_compute_statistics(mt_feats_norm_or.T, cls)
            hmm = hmmlearn.hmm.GaussianHMM(start_prob.shape[0], "diag")
            hmm.startprob_ = start_prob
            hmm.transmat_ = transmat            
            hmm.means_ = means; hmm.covars_ = cov
            cls = hmm.predict(mt_feats_norm_or)                        
    # Post-process method 2: median filtering:
    cls = scipy.signal.medfilt(cls, 5)

    class_names = ["speaker{0:d}".format(c) for c in range(num_speakers)]

    # load ground-truth if available
    gt_file = filename.replace('.wav', '.segments')
    # if groundtruth exists
    if os.path.isfile(gt_file):
        seg_start, seg_end, seg_labs = read_segmentation_gt(gt_file)
        flags_gt, class_names_gt = segments_to_labels(seg_start, seg_end,
                                                      seg_labs, mid_step)

    if plot_res:
        fig = plt.figure()    
        if n_speakers > 0:
            ax1 = fig.add_subplot(111)
        else:
            ax1 = fig.add_subplot(211)
        ax1.set_yticks(np.array(range(len(class_names))))
        ax1.axis((0, duration, -1, len(class_names)))
        ax1.set_yticklabels(class_names)
        ax1.plot(np.array(range(len(cls))) * mid_step + mid_step / 2.0, cls)

    purity_cluster_m, purity_speaker_m = -1, -1
    if os.path.isfile(gt_file):
        if plot_res:
            ax1.plot(np.array(range(len(flags_gt))) *
                     mid_step + mid_step / 2.0, flags_gt, 'r')
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
        if n_speakers <= 0:
            plt.subplot(212)
            plt.plot(s_range, sil_all)
            plt.xlabel("number of clusters")
            plt.ylabel("average clustering's sillouette")
        plt.show()
    return cls, purity_cluster_m, purity_speaker_m


def speaker_diarization_evaluation(folder_name, lda_dimensions):
    """
        This function prints the cluster purity and speaker purity for
        each WAV file stored in a provided directory (.SEGMENT files
         are needed as ground-truth)
        ARGUMENTS:
            - folder_name:     the full path of the folder where the WAV and
                               segment (ground-truth) files are stored
            - lda_dimensions:  a list of LDA dimensions (0 for no LDA)
    """
    types = ('*.wav', )
    wav_files = []
    for files in types:
        wav_files.extend(glob.glob(os.path.join(folder_name, files)))
    
    wav_files = sorted(wav_files)

    # get number of unique speakers per file (from ground-truth)    
    num_speakers = []
    for wav_file in wav_files:
        gt_file = wav_file.replace('.wav', '.segments')
        if os.path.isfile(gt_file):
            _, _, seg_labs = read_segmentation_gt(gt_file)
            num_speakers.append(len(list(set(seg_labs))))
        else:
            num_speakers.append(-1)
    
    for dim in lda_dimensions:
        print("LDA = {0:d}".format(dim))
        for i, wav_file in enumerate(wav_files):
            speaker_diarization(wav_file, num_speakers[i], 2.0, 0.2, 0.05, dim,
                                plot_res=False)


def music_thumbnailing(signal, sampling_rate, short_window=1.0, short_step=0.5,
                       thumb_size=10.0, limit_1=0, limit_2=1):
    """
    This function detects instances of the most representative part of a
    music recording, also called "music thumbnails".
    A technique similar to the one proposed in [1], however a wider set of
    audio features is used instead of chroma features.
    In particular the following steps are followed:
     - Extract short-term audio features. Typical short-term window size: 1
       second
     - Compute the self-similarity matrix, i.e. all pairwise similarities
       between feature vectors
     - Apply a diagonal mask is as a moving average filter on the values of the
       self-similarty matrix.
       The size of the mask is equal to the desirable thumbnail length.
     - Find the position of the maximum value of the new (filtered)
       self-similarity matrix. The audio segments that correspond to the
       diagonial around that position are the selected thumbnails
    

    ARGUMENTS:
     - signal:            input signal
     - sampling_rate:            sampling frequency
     - short_window:     window size (in seconds)
     - short_step:    window step (in seconds)
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
    signal = audioBasicIO.stereo_to_mono(signal)
    # feature extraction:
    st_feats, _ = stf.feature_extraction(signal, sampling_rate,
                                         sampling_rate * short_window,
                                         sampling_rate * short_step)

    # self-similarity matrix
    sim_matrix = self_similarity_matrix(st_feats)

    # moving filter:
    m_filter = int(round(thumb_size / short_step))
    diagonal = np.eye(m_filter, m_filter)
    sim_matrix = scipy.signal.convolve2d(sim_matrix, diagonal, 'valid')

    # post-processing (remove main diagonal elements)
    min_sm = np.min(sim_matrix)
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            if abs(i-j) < 5.0 / short_step or i > j:
                sim_matrix[i, j] = min_sm

    # find max position:
    sim_matrix[0:int(limit_1 * sim_matrix.shape[0]), :] = min_sm
    sim_matrix[:, 0:int(limit_1 * sim_matrix.shape[0])] = min_sm
    sim_matrix[int(limit_2 * sim_matrix.shape[0])::, :] = min_sm
    sim_matrix[:, int(limit_2 * sim_matrix.shape[0])::] = min_sm

    rows, cols = np.unravel_index(sim_matrix.argmax(), sim_matrix.shape)
    i1 = rows
    i2 = rows
    j1 = cols
    j2 = cols

    while i2-i1 < m_filter:
        if i1 <= 0 or j1 <= 0 or i2 >= sim_matrix.shape[0]-2 or \
                j2 >= sim_matrix.shape[1]-2:
            break
        if sim_matrix[i1-1, j1-1] > sim_matrix[i2 + 1, j2 + 1]:
            i1 -= 1
            j1 -= 1            
        else:            
            i2 += 1
            j2 += 1            

    return short_step * i1, short_step * i2, short_step * j1, short_step * j2, \
        sim_matrix



