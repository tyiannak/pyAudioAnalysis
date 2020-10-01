from __future__ import print_function
import sys
import numpy as np
import os
import glob
import pickle as cPickle
import csv
import ntpath
from pyAudioAnalysis import MidTermFeatures as aF
from pyAudioAnalysis import audioBasicIO
from scipy import linalg as la
from scipy.spatial import distance
import sklearn.svm
import sklearn.decomposition
import sklearn.ensemble
import plotly
import plotly.graph_objs as go
import sklearn.metrics


shortTermWindow = 0.050
shortTermStep = 0.050
eps = 0.00000001


class Knn:
    def __init__(self, features, labels, neighbors):
        self.features = features
        self.labels = labels
        self.neighbors = neighbors

    def classify(self, test_sample):
        n_classes = np.unique(self.labels).shape[0]
        y_dist = (distance.cdist(self.features,
                                 test_sample.reshape(1, test_sample.shape[0]),
                                 'euclidean')).T
        i_sort = np.argsort(y_dist)
        P = np.zeros((n_classes,))
        for i in range(n_classes):
            P[i] = np.nonzero(self.labels[i_sort[0]
                [0:self.neighbors]] == i)[0].shape[0] / float(self.neighbors)
        return np.argmax(P), P


def classifier_wrapper(classifier, classifier_type, test_sample):
    """
    This function is used as a wrapper to pattern classification.
    ARGUMENTS:
        - classifier:        a classifier object of type sklearn.svm.SVC or 
                             kNN (defined in this library) or sklearn.ensemble.
                             RandomForestClassifier or sklearn.ensemble.
                             GradientBoostingClassifier  or 
                             sklearn.ensemble.ExtraTreesClassifier
        - classifier_type:   "svm" or "knn" or "randomforests" or 
                             "gradientboosting" or "extratrees"
        - test_sample:        a feature vector (np array)
    RETURNS:
        - R:            class ID
        - P:            probability estimate

    EXAMPLE (for some audio signal stored in array x):
        import audioFeatureExtraction as aF
        import audioTrainTest as aT
        # load the classifier (here SVM, for kNN use load_model_knn instead):
        [classifier, MEAN, STD, classNames, mt_win, mt_step, st_win, st_step] =
        aT.load_model(model_name)
        # mid-term feature extraction:
        [mt_features, _, _] = aF.mid_feature_extraction(x, Fs, mt_win * Fs,
        mt_step * Fs, round(Fs*st_win), round(Fs*st_step));
        # feature normalization:
        curFV = (mt_features[:, i] - MEAN) / STD;
        # classification
        [Result, P] = classifierWrapper(classifier, model_type, curFV)
    """
    class_id = -1
    probability = -1
    if classifier_type == "knn":
        class_id, probability = classifier.classify(test_sample)
    elif classifier_type == "svm" or \
            classifier_type == "randomforest" or \
            classifier_type == "gradientboosting" or \
            classifier_type == "extratrees" or \
            classifier_type == "svm_rbf":
        class_id = classifier.predict(test_sample.reshape(1, -1))[0]
        probability = classifier.predict_proba(test_sample.reshape(1, -1))[0]
    return class_id, probability


def regression_wrapper(model, model_type, test_sample):
    """
    This function is used as a wrapper to pattern classification.
    ARGUMENTS:
        - model:        regression model
        - model_type:        "svm" or "knn" (TODO)
        - test_sample:        a feature vector (np array)
    RETURNS:
        - R:            regression result (estimated value)

    EXAMPLE (for some audio signal stored in array x):
        TODO
    """
    if model_type == "svm" or model_type == "randomforest" or \
            model_type == "svm_rbf":
        return model.predict(test_sample.reshape(1,-1))[0]

    #    elif classifier_type == "knn":
    #    TODO


def random_split_features(features, percentage):
    """
    def randSplitFeatures(features):

    This function splits a feature set for training and testing.

    ARGUMENTS:
        - features:         a list ([numOfClasses x 1]) whose elements 
                            containt np matrices of features.
                            each matrix features[i] of class i is 
                            [n_samples x numOfDimensions]
        - per_train:        percentage
    RETURNS:
        - featuresTrains:   a list of training data for each class
        - f_test:           a list of testing data for each class
    """

    f_train = []
    f_test = []
    for index, feat in enumerate(features):
        n_samples, _ = feat.shape
        randperm = np.random.permutation(range(n_samples))
        n_train = int(round(percentage * n_samples))
        f_train.append(feat[randperm[0:n_train]])
        f_test.append(feat[randperm[n_train::]])
    return f_train, f_test


def train_knn(features, neighbors):
    """
    Train a kNN  classifier.
    ARGUMENTS:
        - features:         a list ([numOfClasses x 1]) whose elements 
                            contain np matrices of features.
                            each matrix features[i] of class i is 
                            [n_samples x numOfDimensions]
        - neighbors:                parameter K
    RETURNS:
        - kNN:              the trained kNN variable

    """
    feature_matrix, labels = features_to_matrix(features)
    knn = Knn(feature_matrix, labels, neighbors)
    return knn


def train_svm(features, c_param, kernel='linear'):
    """
    Train a multi-class probabilitistic SVM classifier.
    Note:     This function is simply a wrapper to the sklearn functionality 
              for SVM training
              See function trainSVM_feature() to use a wrapper on both the 
              feature extraction and the SVM training
              (and parameter tuning) processes.
    ARGUMENTS:
        - features:         a list ([numOfClasses x 1]) whose elements 
                            containt np matrices of features  each matrix 
                            features[i] of class i is 
                            [n_samples x numOfDimensions]
        - c_param:           SVM parameter C (cost of constraints violation)
    RETURNS:
        - svm:              the trained SVM variable

    NOTE:
        This function trains a linear-kernel SVM for a given C value.
        For a different kernel, other types of parameters should be provided.
    """

    feature_matrix, labels = features_to_matrix(features)
    svm = sklearn.svm.SVC(C=c_param, kernel=kernel, probability=True,
                          gamma='auto')
    svm.fit(feature_matrix, labels)

    return svm


def train_random_forest(features, n_estimators):
    """
    Train a multi-class random forest classifier.
    Note:     This function is simply a wrapper to the sklearn functionality
              for model training.
              See function extract_features_and_train() to use a wrapper on both
              the feature extraction and the model training (and parameter
              tuning) processes.
    ARGUMENTS:
        - features:         a list ([numOfClasses x 1]) whose elements
                            containt np matrices of features
                            each matrix features[i] of class i is
                            [n_samples x numOfDimensions]
        - n_estimators:     number of trees in the forest
    RETURNS:
        - rf:               the trained random forest

    """

    feature_matrix, labels = features_to_matrix(features)
    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators)
    rf.fit(feature_matrix, labels)

    return rf


def train_gradient_boosting(features, n_estimators):
    """
    Train a gradient boosting classifier
    Note:     This function is simply a wrapper to the sklearn functionality
              for model training.
              See function extract_features_and_train() to use a wrapper on both
              the feature extraction and the model training (and parameter
              tuning) processes.
    ARGUMENTS:
        - features:         a list ([numOfClasses x 1]) whose elements containt
                            np matrices of features. each matrix features[i]
                            of class i is [n_samples x numOfDimensions]
        - n_estimators:     number of trees in the forest
    RETURNS:
        - rf:              the trained model
    """

    feature_matrix, labels = features_to_matrix(features)
    rf = sklearn.ensemble.GradientBoostingClassifier(n_estimators=n_estimators)
    rf.fit(feature_matrix, labels)

    return rf


def train_extra_trees(features, n_estimators):
    """
    Train an extra tree
    Note:     This function is simply a wrapper to the sklearn functionality
              for model training.
              See function extract_features_and_train() to use a wrapper on both
              the feature extraction and the model training (and parameter
              tuning) processes.
    ARGUMENTS:
        - features:         a list ([numOfClasses x 1]) whose elements
                            containt np matrices of features
                            each matrix features[i] of class i is
                            [n_samples x numOfDimensions]
        - n_estimators:     number of trees in the forest
    RETURNS:
        - et:               the trained model
    """

    feature_matrix, labels = features_to_matrix(features)
    et = sklearn.ensemble.ExtraTreesClassifier(n_estimators=n_estimators)
    et.fit(feature_matrix,labels)

    return et


def train_svm_regression(features, labels, c_param, kernel='linear'):
    svm = sklearn.svm.SVR(C=c_param, kernel=kernel)
    svm.fit(features, labels)
    train_err = np.mean(np.abs(svm.predict(features) - labels))
    return svm, train_err


def train_random_forest_regression(features, labels, n_estimators):
    rf = sklearn.ensemble.RandomForestRegressor(n_estimators=n_estimators)
    rf.fit(features, labels)
    train_err = np.mean(np.abs(rf.predict(features) - labels))
    return rf, train_err


def extract_features_and_train(paths, mid_window, mid_step, short_window,
                               short_step, classifier_type, model_name,
                               compute_beat=False, train_percentage=0.90):
    """
    This function is used as a wrapper to segment-based audio feature extraction
    and classifier training.
    ARGUMENTS:
        paths:                      list of paths of directories. Each directory
                                    contains a signle audio class whose samples
                                    are stored in seperate WAV files.
        mid_window, mid_step:       mid-term window length and step
        short_window, short_step:   short-term window and step
        classifier_type:            "svm" or "knn" or "randomforest" or
                                    "gradientboosting" or "extratrees"
        model_name:                 name of the model to be saved
    RETURNS:
        None. Resulting classifier along with the respective model
        parameters are saved on files.
    """

    # STEP A: Feature Extraction:
    features, class_names, _ = \
        aF.multiple_directory_feature_extraction(paths, mid_window, mid_step,
                                                 short_window, short_step,
                                                 compute_beat=compute_beat)

    if len(features) == 0:
        print("trainSVM_feature ERROR: No data found in any input folder!")
        return

    n_feats = features[0].shape[1]
    feature_names = ["features" + str(d + 1) for d in range(n_feats)]

    write_train_data_arff(model_name, features, class_names, feature_names)

    for i, feat in enumerate(features):
        if len(feat) == 0:
            print("trainSVM_feature ERROR: " + paths[i] +
                  " folder is empty or non-existing!")
            return

    # STEP B: classifier Evaluation and Parameter Selection:
    if classifier_type == "svm" or classifier_type == "svm_rbf":
        classifier_par = np.array([0.001, 0.01,  0.5, 1.0, 5.0, 10.0, 20.0])
    elif classifier_type == "randomforest":
        classifier_par = np.array([10, 25, 50, 100, 200, 500])
    elif classifier_type == "knn":
        classifier_par = np.array([1, 3, 5, 7, 9, 11, 13, 15])        
    elif classifier_type == "gradientboosting":
        classifier_par = np.array([10, 25, 50, 100, 200, 500])
    elif classifier_type == "extratrees":
        classifier_par = np.array([10, 25, 50, 100, 200, 500])

    # get optimal classifeir parameter:
    temp_features = []
    for feat in features:
        temp = []
        for i in range(feat.shape[0]):
            temp_fv = feat[i, :]
            if (not np.isnan(temp_fv).any()) and (not np.isinf(temp_fv).any()):
                temp.append(temp_fv.tolist())
            else:
                print("NaN Found! Feature vector not used for training")
        temp_features.append(np.array(temp))
    features = temp_features

    best_param = evaluate_classifier(features, class_names, 100, classifier_type,
                                     classifier_par, 0, train_percentage)

    print("Selected params: {0:.5f}".format(best_param))

    features_norm, mean, std = normalize_features(features)
    mean = mean.tolist()
    std = std.tolist()

    # STEP C: Save the classifier to file
    if classifier_type == "svm":
        classifier = train_svm(features_norm, best_param)
    elif classifier_type == "svm_rbf":
        classifier = train_svm(features_norm, best_param, kernel='rbf')
    elif classifier_type == "randomforest":
        classifier = train_random_forest(features_norm, best_param)
    elif classifier_type == "gradientboosting":
        classifier = train_gradient_boosting(features_norm, best_param)
    elif classifier_type == "extratrees":
        classifier = train_extra_trees(features_norm, best_param)

    if classifier_type == "knn":
        feature_matrix, labels = features_to_matrix(features_norm)
        feature_matrix = feature_matrix.tolist()
        labels = labels.tolist()
        save_path = model_name
        save_parameters(save_path, feature_matrix, labels, mean, std,
                        class_names, best_param, mid_window, mid_step,
                        short_window, short_step, compute_beat)

    elif classifier_type == "svm" or classifier_type == "svm_rbf" or \
            classifier_type == "randomforest" or \
            classifier_type == "gradientboosting" or \
            classifier_type == "extratrees":
        with open(model_name, 'wb') as fid:
            cPickle.dump(classifier, fid)
        save_path = model_name + "MEANS"
        save_parameters(save_path, mean, std, class_names, mid_window, mid_step,
                        short_window, short_step, compute_beat)


def save_parameters(path, *parameters):
    with open(path, 'wb') as file_handle:
        for param in parameters:
            cPickle.dump(param, file_handle, protocol=cPickle.HIGHEST_PROTOCOL)


def feature_extraction_train_regression(folder_name, mid_window, mid_step,
                                        short_window, short_step, model_type,
                                        model_name, compute_beat=False):
    """
    This function is used as a wrapper to segment-based audio
    feature extraction and classifier training.
    ARGUMENTS:
        folder_name:        path of directory containing the WAV files
                         and Regression CSVs
        mt_win, mt_step:        mid-term window length and step
        st_win, st_step:        short-term window and step
        model_type:        "svm" or "knn" or "randomforest"
        model_name:        name of the model to be saved
    RETURNS:
        None. Resulting regression model along with the respective
        model parameters are saved on files.
    """
    # STEP A: Feature Extraction:
    features, _, filenames = \
        aF.multiple_directory_feature_extraction([folder_name], mid_window,
                                                 mid_step, short_window,
                                                 short_step,
                                                 compute_beat=compute_beat)
    features = features[0]
    filenames = [ntpath.basename(f) for f in filenames[0]]
    f_final = []

    # Read CSVs:
    csv_files = glob.glob(folder_name + os.sep + "*.csv")
    regression_labels = []
    regression_names = []
    f_final = []
    for c in csv_files:
        cur_regression_labels = []
        f_temp = []
        # open the csv file that contains the current target value's annotations
        with open(c, 'rt') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in csv_reader:
                if len(row) == 2:
                    # ... and if the current filename exists
                    # in the list of filenames
                    if row[0] in filenames:
                        index = filenames.index(row[0])
                        cur_regression_labels.append(float(row[1]))
                        f_temp.append(features[index, :])
                    else:
                        print("Warning: {} not found "
                              "in list of files.".format(row[0]))
                else:
                    print("Warning: Row with unknown format in regression file")

        f_final.append(np.array(f_temp))
        # cur_regression_labels is the list of values
        # for the current regression problem
        regression_labels.append(np.array(cur_regression_labels))
        # regression task name
        regression_names.append(ntpath.basename(c).replace(".csv", ""))
        if len(features) == 0:
            print("ERROR: No data found in any input folder!")
            return

    # TODO: ARRF WRITE????
    # STEP B: classifier Evaluation and Parameter Selection:
    if model_type == "svm" or model_type == "svm_rbf":
        model_params = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5,
                                 1.0, 5.0, 10.0])
    elif model_type == "randomforest":
        model_params = np.array([5, 10, 25, 50, 100])

    errors = []
    errors_base = []
    best_params = []

    for iRegression, r in enumerate(regression_names):
        # get optimal classifeir parameter:
        print("Regression task " + r)
        bestParam, error, berror = evaluate_regression(f_final[iRegression],
                                                       regression_labels[
                                                          iRegression],
                                                       100, model_type,
                                                       model_params)
        errors.append(error)
        errors_base.append(berror)
        best_params.append(bestParam)
        print("Selected params: {0:.5f}".format(bestParam))

        features_norm, mean, std = normalize_features([f_final[iRegression]])

        # STEP C: Save the model to file
        if model_type == "svm":
            classifier, _ = train_svm_regression(features_norm[0],
                                                 regression_labels[iRegression],
                                                 bestParam)
        if model_type == "svm_rbf":
            classifier, _ = train_svm_regression(features_norm[0],
                                                 regression_labels[iRegression],
                                                 bestParam, kernel='rbf')
        if model_type == "randomforest":
            classifier, _ = train_random_forest_regression(features_norm[0],
                                                           regression_labels[
                                                            iRegression],
                                                           bestParam)

        if model_type == "svm" or model_type == "svm_rbf" \
                or model_type == "randomforest":
            with open(model_name + "_" + r, 'wb') as fid:
                cPickle.dump(classifier, fid)
            save_path = model_name + "_" + r + "MEANS"
            save_parameters(save_path, mean, std, mid_window, mid_step,
                            short_window, short_step, compute_beat)

    return errors, errors_base, best_params


def load_model_knn(knn_model_name, is_regression=False):
    with open(knn_model_name, "rb") as fo:
        features = cPickle.load(fo)
        labels = cPickle.load(fo)
        mean = cPickle.load(fo)
        std = cPickle.load(fo)
        if not is_regression:
            classes = cPickle.load(fo)
        neighbors = cPickle.load(fo)
        mid_window = cPickle.load(fo)
        mid_step = cPickle.load(fo)
        short_window = cPickle.load(fo)
        short_step = cPickle.load(fo)
        compute_beat = cPickle.load(fo)

    features = np.array(features)
    labels = np.array(labels)
    mean = np.array(mean)
    std = np.array(std)

    classifier = Knn(features, labels, neighbors)
    # Note: a direct call to the kNN constructor is used here

    if is_regression:
        return classifier, mean, std, mid_window, mid_step, short_window, \
               short_step, compute_beat
    else:
        return classifier, mean, std, classes, mid_window, mid_step, \
               short_window, short_step, compute_beat


def load_model(model_name, is_regression=False):
    """
    This function loads an SVM model either for classification or training.
    ARGMUMENTS:
        - SVMmodel_name:     the path of the model to be loaded
        - is_regression:     a flag indigating whereas this model
                             is regression or not
    """
    with open(model_name + "MEANS", "rb") as fo:
        mean = cPickle.load(fo)
        std = cPickle.load(fo)
        if not is_regression:
            classNames = cPickle.load(fo)
        mid_window = cPickle.load(fo)
        mid_step = cPickle.load(fo)
        short_window = cPickle.load(fo)
        short_step = cPickle.load(fo)
        compute_beat = cPickle.load(fo)

    mean = np.array(mean)
    std = np.array(std)

    with open(model_name, 'rb') as fid:
        svm_model = cPickle.load(fid)

    if is_regression:
        return svm_model, mean, std, mid_window, mid_step, short_window, \
               short_step, compute_beat
    else:
        return svm_model, mean, std, classNames, mid_window, mid_step, \
               short_window, short_step, compute_beat


def evaluate_classifier(features, class_names, n_exp, classifier_name, params,
                        parameter_mode, train_percentage=0.90):
    """
    ARGUMENTS:
        features:     a list ([numOfClasses x 1]) whose elements containt
                      np matrices of features. Each matrix features[i] of
                      class i is [n_samples x numOfDimensions]
        class_names:    list of class names (strings)
        n_exp:        number of cross-validation experiments
        classifier_name: svm or knn or randomforest
        params:        list of classifier parameters (for parameter
                       tuning during cross-validation)
        parameter_mode:    0: choose parameters that lead to maximum overall
                             classification ACCURACY
                          1: choose parameters that lead to maximum overall
                          f1 MEASURE
    RETURNS:
         bestParam:    the value of the input parameter that optimizes the
         selected performance measure
    """

    # feature normalization:
    features_norm, MEAN, STD = normalize_features(features)
    # features_norm = features;
    n_classes = len(features)
    ac_all = []
    f1_all = []
    pre_class_all = []
    rec_classes_all = []
    f1_classes_all = []
    cms_all = []

    # compute total number of samples:
    n_samples_total = 0
    for f in features:
        n_samples_total += f.shape[0]
    if n_samples_total > 10000 and n_exp > 2:
        n_exp = 2
        print("Number of training experiments changed to 2 due to "
              "very high number of samples")
    elif n_samples_total > 2000 and n_exp > 5:
        n_exp = 5
        print("Number of training experiments changed to 5 due to "
              "high number of samples")
    elif n_samples_total > 1000 and n_exp > 10:
        n_exp = 10
        print("Number of training experiments changed to 10 due to "
              "high number of samples")

    for Ci, C in enumerate(params):
        # for each param value
        cm = np.zeros((n_classes, n_classes))
        for e in range(n_exp):
            # for each cross-validation iteration:
            print("Param = {0:.5f} - classifier Evaluation "
                  "Experiment {1:d} of {2:d}".format(C, e+1, n_exp))
            # split features:
            f_train, f_test = random_split_features(features_norm,
                                                    train_percentage)
            # train multi-class svms:
            if classifier_name == "svm":
                classifier = train_svm(f_train, C)
            elif classifier_name == "svm_rbf":
                classifier = train_svm(f_train, C, kernel='rbf')
            elif classifier_name == "knn":
                classifier = train_knn(f_train, C)
            elif classifier_name == "randomforest":
                classifier = train_random_forest(f_train, C)
            elif classifier_name == "gradientboosting":
                classifier = train_gradient_boosting(f_train, C)
            elif classifier_name == "extratrees":
                classifier = train_extra_trees(f_train, C)

            cmt = np.zeros((n_classes, n_classes))
            for c1 in range(n_classes):
                n_test_samples = len(f_test[c1])
                res = np.zeros((n_test_samples, 1))
                for ss in range(n_test_samples):
                    res[ss], _ = classifier_wrapper(classifier,
                                                    classifier_name,
                                                    f_test[c1][ss])
                for c2 in range(n_classes):
                    cmt[c1][c2] = float(len(np.nonzero(res == c2)[0]))
            cm = cm + cmt
        cm = cm + 0.0000000010
        rec = np.zeros((cm.shape[0], ))
        pre = np.zeros((cm.shape[0], ))

        for ci in range(cm.shape[0]):
            rec[ci] = cm[ci, ci] / np.sum(cm[ci, :])
            pre[ci] = cm[ci, ci] / np.sum(cm[:, ci])
        pre_class_all.append(pre)
        rec_classes_all.append(rec)
        f1 = 2 * rec * pre / (rec + pre)
        f1_classes_all.append(f1)
        ac_all.append(np.sum(np.diagonal(cm)) / np.sum(cm))

        cms_all.append(cm)
        f1_all.append(np.mean(f1))

    print("\t\t", end="")
    for i, c in enumerate(class_names):
        if i == len(class_names)-1:
            print("{0:s}\t\t".format(c), end="")
        else:
            print("{0:s}\t\t\t".format(c), end="")
    print("OVERALL")
    print("\tC", end="")
    for c in class_names:
        print("\tPRE\tREC\tf1", end="")
    print("\t{0:s}\t{1:s}".format("ACC", "f1"))
    best_ac_ind = np.argmax(ac_all)
    best_f1_ind = np.argmax(f1_all)
    for i in range(len(pre_class_all)):
        print("\t{0:.3f}".format(params[i]), end="")
        for c in range(len(pre_class_all[i])):
            print("\t{0:.1f}\t{1:.1f}\t{2:.1f}".format(100.0 *
                                                       pre_class_all[i][c],
                                                       100.0 *
                                                       rec_classes_all[i][c],
                                                       100.0 *
                                                       f1_classes_all[i][c]),
                  end="")
        print("\t{0:.1f}\t{1:.1f}".format(100.0 * ac_all[i], 100.0 * f1_all[i]),
              end="")
        if i == best_f1_ind:
            print("\t best f1", end="")
        if i == best_ac_ind:
            print("\t best Acc", end="")
        print("")

    if parameter_mode == 0:
        # keep parameters that maximize overall classification accuracy:
        print("Confusion Matrix:")
        print_confusion_matrix(cms_all[best_ac_ind], class_names)
        return params[best_ac_ind]
    elif parameter_mode == 1:
        # keep parameters that maximize overall f1 measure:
        print("Confusion Matrix:")
        print_confusion_matrix(cms_all[best_f1_ind], class_names)
        return params[best_f1_ind]


def evaluate_regression(features, labels, n_exp, method_name, params):
    """
    ARGUMENTS:
        features:     np matrices of features [n_samples x numOfDimensions]
        labels:       list of sample labels
        n_exp:         number of cross-validation experiments
        method_name:   "svm" or "randomforest"
        params:       list of classifier params to be evaluated
    RETURNS:
         bestParam:   the value of the input parameter that optimizes
         the selected performance measure
    """

    # feature normalization:
    features_norm, mean, std = normalize_features([features])
    features_norm = features_norm[0]
    n_samples = labels.shape[0]
    per_train = 0.9
    errors_all = []
    er_train_all = []
    er_base_all = []
    for Ci, C in enumerate(params):   # for each param value
                errors = []
                errors_train = []
                errors_baseline = []
                for e in range(n_exp):   # for each cross-validation iteration:
                    # split features:
                    randperm = np.random.permutation(range(n_samples))
                    n_train = int(round(per_train * n_samples))
                    f_train = [features_norm[randperm[i]]
                               for i in range(n_train)]
                    f_test = [features_norm[randperm[i+n_train]]
                              for i in range(n_samples - n_train)]
                    l_train = [labels[randperm[i]] for i in range(n_train)]
                    l_test = [labels[randperm[i + n_train]]
                              for i in range(n_samples - n_train)]

                    # train multi-class svms:                    
                    f_train = np.matrix(f_train)                                 
                    if method_name == "svm":                                        
                        classifier, train_err = \
                            train_svm_regression(f_train, l_train, C)
                    elif method_name == "svm_rbf":                      
                        classifier, train_err = \
                            train_svm_regression(f_train, l_train, C,
                                                 kernel='rbf')
                    elif method_name == "randomforest":
                        classifier, train_err = \
                            train_random_forest_regression(f_train, l_train, C)
                    error_test = []
                    error_test_baseline = []
                    for itest, fTest in enumerate(f_test):
                        R = regression_wrapper(classifier, method_name, fTest)
                        Rbaseline = np.mean(l_train)
                        error_test.append((R - l_test[itest]) *
                                          (R - l_test[itest]))
                        error_test_baseline.append((Rbaseline - l_test[itest]) *
                                                  (Rbaseline - l_test[itest]))
                    error = np.array(error_test).mean()
                    error_baseline = np.array(error_test_baseline).mean()
                    errors.append(error)
                    errors_train.append(train_err)
                    errors_baseline.append(error_baseline)
                errors_all.append(np.array(errors).mean())
                er_train_all.append(np.array(errors_train).mean())
                er_base_all.append(np.array(errors_baseline).mean())

    best_ind = np.argmin(errors_all)

    print("{0:s}\t\t{1:s}\t\t{2:s}\t\t{3:s}".format("Param", "MSE",
                                                    "T-MSE", "R-MSE"))
    for i in range(len(errors_all)):
        print("{0:.4f}\t\t{1:.2f}\t\t{2:.2f}\t\t{3:.2f}".format(params[i],
                                                                errors_all[i],
                                                                er_train_all[i],
                                                                er_base_all[i]),
              end="")
        if i == best_ind:
            print("\t\t best",end="")
        print("")
    return params[best_ind], errors_all[best_ind], er_base_all[best_ind]


def print_confusion_matrix(cm, class_names):
    """
    This function prints a confusion matrix for a particular classification task.
    ARGUMENTS:
        cm:            a 2-D np array of the confusion matrix
                       (cm[i,j] is the number of times a sample from class i
                       was classified in class j)
        class_names:    a list that contains the names of the classes
    """

    if cm.shape[0] != len(class_names):
        print("printConfusionMatrix: Wrong argument sizes\n")
        return

    for c in class_names:
        if len(c) > 4:
            c = c[0:3]
        print("\t{0:s}".format(c), end="")
    print("")

    for i, c in enumerate(class_names):
        if len(c) > 4:
            c = c[0:3]
        print("{0:s}".format(c), end="")
        for j in range(len(class_names)):
            print("\t{0:.2f}".format(100.0 * cm[i][j] / np.sum(cm)), end="")
        print("")


def normalize_features(features):
    """
    This function normalizes a feature set to 0-mean and 1-std.
    Used in most classifier trainning cases.

    ARGUMENTS:
        - features:    list of feature matrices (each one of them is a np
                       matrix)
    RETURNS:
        - features_norm:    list of NORMALIZED feature matrices
        - mean:        mean vector
        - std:        std vector
    """
    temp_feats = np.array([])

    for count, f in enumerate(features):
        if f.shape[0] > 0:
            if count == 0:
                temp_feats = f
            else:
                temp_feats = np.vstack((temp_feats, f))
            count += 1

    mean = np.mean(temp_feats, axis=0) + 1e-14
    std = np.std(temp_feats, axis=0) + 1e-14

    features_norm = []
    for f in features:
        ft = f.copy()
        for n_samples in range(f.shape[0]):
            ft[n_samples, :] = (ft[n_samples, :] - mean) / std
        features_norm.append(ft)
    return features_norm, mean, std


def features_to_matrix(features):
    """
    features_to_matrix(features)

    This function takes a list of feature matrices as argument and returns
    a single concatenated feature matrix and the respective class labels.

    ARGUMENTS:
        - features:        a list of feature matrices

    RETURNS:
        - feature_matrix:    a concatenated matrix of features
        - labels:            a vector of class indices
    """

    labels = np.array([])
    feature_matrix = np.array([])
    for i, f in enumerate(features):
        if i == 0:
            feature_matrix = f
            labels = i * np.ones((len(f), 1))
        else:
            feature_matrix = np.vstack((feature_matrix, f))
            labels = np.append(labels, i * np.ones((len(f), 1)))
    return feature_matrix, labels


def pca_wrapper(features, dimensions):
    features, labels = features_to_matrix(features)
    pca = sklearn.decomposition.PCA(n_components = dimensions)
    pca.fit(features)
    coeff = pca.components_
    coeff = coeff[:, 0:dimensions]

    features_transformed = []
    for f in features:
        ft = f.copy()
        # ft = pca.transform(ft, k=nDims)
        ft = np.dot(f, coeff)
        features_transformed.append(ft)

    return features_transformed, coeff


def compute_class_rec_pre_f1(c_mat):
    """
    Gets recall, precision and f1 PER CLASS, given the confusion matrix
    :param c_mat: the [n_class x n_class] confusion matrix
    :return: rec, pre and f1 for each class
    """
    n_class = c_mat.shape[0]
    rec, pre, f1 = [], [], []
    for i in range(n_class):
        rec.append(float(c_mat[i, i]) / np.sum(c_mat[i, :]))
        pre.append(float(c_mat[i, i]) / np.sum(c_mat[:, i]))
        f1.append(2 * rec[-1] * pre[-1] / (rec[-1] + pre[-1]))
    return rec,  pre, f1


def evaluate_model_for_folders(input_test_folders, model_name, model_type,
                               positive_class, plot=True):
    """
    evaluate_model_for_folders(input_test_folders, model_name, model_type)
    This function evaluates a model by computing the confusion matrix, the
    per class performance metrics and by generating a ROC and Precision / Recall
    diagrams (for a particular class of interest), for a given test dataset.
    The dataset needs to be organized in folders (one folder per audio class),
    exactly like in extract_features_and_train()
    :param input_test_folders:  list of folders (each folder represents a
    separate audio class)
    :param model_name:  path to the model to be tested
    :param model_type:  type of the model
    :param positive_class name of the positive class
    :param plot (True default) if to plot 2 diagrams on plotly
    :return: thr_prre, pre, rec  (thresholds, precision recall values)
    thr_roc, fpr, tpr (thresholds, false positive , true positive rates)

    Usage example:
    from pyAudioAnalysis import audioTrainTest as aT
    thr_prre, pre, rec, thr_roc, fpr, tpr =
    aT.evaluate_model_for_folders(["4_classes_small/speech",
                                   "4_classes_small/music"],
                                   "data/models/svm_rbf_4class",
                                   "svm_rbf", "speech")
    """
    
    class_names = []
    y_true_binary = []
    y_true = []
    y_pred = []
    probs_positive = []
    for i, d in enumerate(input_test_folders):
        if d[-1] == os.sep:
            class_names.append(d.split(os.sep)[-2])
        else:
            class_names.append(d.split(os.sep)[-1])

        types = ('*.wav', '*.aif', '*.aiff', '*.mp3', '*.au', '*.ogg')
        wav_file_list = []
        for files in types:
            wav_file_list.extend(glob.glob(os.path.join(d, files)))
        # get list of audio files for current folder and run classifier
        for w in wav_file_list:
            c, p, probs_names = file_classification(w, model_name, model_type)
            y_pred.append(c)
            y_true.append(probs_names.index(class_names[i]))
            if i==probs_names.index(positive_class):
                y_true_binary.append(1)
            else:
                y_true_binary.append(0)

            prob_positive = p[probs_names.index(positive_class)]
            probs_positive.append(prob_positive)

    pre, rec, thr_prre = sklearn.metrics.precision_recall_curve(y_true_binary,
                                                                probs_positive)
    fpr, tpr, thr_roc = sklearn.metrics.roc_curve(y_true_binary, probs_positive)
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    rec_c,  pre_c, f1_c = compute_class_rec_pre_f1(cm)
    f1 = (sklearn.metrics.f1_score(y_true, y_pred, average='micro'))
    acc = (sklearn.metrics.accuracy_score(y_true, y_pred))
    print(cm)
    print(rec_c, pre_c, f1_c, f1, acc)
    if plot:
        titles = ["Confusion matrix, acc = {0:.1f}%, "
                  " F1 (micro): {1:.1f}%".format(100 * acc, 100 * f1),
                  "Class-wise Performance measures",
                  "Pre vs Rec for " + positive_class,
                  "ROC for " + positive_class]
        figs = plotly.subplots.make_subplots(rows=2, cols=2,
                                             subplot_titles=titles)

        heatmap = go.Heatmap(z=np.flip(cm, axis=0), x=class_names,
                             y=list(reversed(class_names)),
                             colorscale=[[0, '#4422ff'], [1, '#ff4422']],
                             name="confusin matrix", showscale=False)
        mark_prop1 = dict(color='rgba(80, 220, 150, 0.5)',
                          line=dict(color='rgba(80, 220, 150, 1)', width=2))
        mark_prop2 = dict(color='rgba(80, 150, 220, 0.5)',
                          line=dict(color='rgba(80, 150, 220, 1)', width=2))
        mark_prop3 = dict(color='rgba(250, 150, 150, 0.5)',
                          line=dict(color='rgba(250, 150, 150, 1)', width=3))
        b1 = go.Bar(x=class_names, y=rec_c, name="Recall", marker=mark_prop1)
        b2 = go.Bar(x=class_names, y=pre_c, name="Precision", marker=mark_prop2)
        b3 = go.Bar(x=class_names, y=f1_c, name="F1", marker=mark_prop3)

        figs.append_trace(heatmap, 1, 1);
        figs.append_trace(b1, 1, 2)
        figs.append_trace(b2, 1, 2);
        figs.append_trace(b3, 1, 2)
        figs.append_trace(go.Scatter(x=thr_prre, y=pre, name="Precision",
                                     marker=mark_prop1), 2, 1)
        figs.append_trace(go.Scatter(x=thr_prre, y=rec, name="Recall",
                                     marker=mark_prop2), 2, 1)
        figs.append_trace(go.Scatter(x=fpr, y=tpr, showlegend=False), 2, 2)
        figs.update_xaxes(title_text="threshold", row=2, col=1)
        figs.update_xaxes(title_text="false positive rate", row=2, col=2)
        figs.update_yaxes(title_text="true positive rate", row=2, col=2)

        plotly.offline.plot(figs, filename="temp.html", auto_open=True)

    return cm, thr_prre, pre, rec, thr_roc, fpr, tpr


def file_classification(input_file, model_name, model_type):
    # Load classifier:
    if not os.path.isfile(model_name):
        print("fileClassification: input model_name not found!")
        return -1, -1, -1

    if isinstance(input_file, str) and not os.path.isfile(input_file):
        print("fileClassification: wav file not found!")
        return -1, -1, -1

    if model_type == 'knn':
        classifier, mean, std, classes, mid_window, mid_step, short_window, \
            short_step, compute_beat = load_model_knn(model_name)
    else:
        classifier, mean, std, classes, mid_window, mid_step, short_window, \
            short_step, compute_beat = load_model(model_name)

    # read audio file and convert to mono
    sampling_rate, signal = audioBasicIO.read_audio_file(input_file)
    signal = audioBasicIO.stereo_to_mono(signal)

    if sampling_rate == 0:
        # audio file IO problem
        return -1, -1, -1
    if signal.shape[0] / float(sampling_rate) < mid_window:
        mid_window = signal.shape[0] / float(sampling_rate)

    # feature extraction:
    mid_features, s, _ = \
        aF.mid_feature_extraction(signal, sampling_rate,
                                  mid_window * sampling_rate,
                                  mid_step * sampling_rate,
                                  round(sampling_rate * short_window),
                                  round(sampling_rate * short_step))
    # long term averaging of mid-term statistics
    mid_features = mid_features.mean(axis=1)
    if compute_beat:
        beat, beat_conf = aF.beat_extraction(s, short_step)
        mid_features = np.append(mid_features, beat)
        mid_features = np.append(mid_features, beat_conf)
    feature_vector = (mid_features - mean) / std    # normalization

    # classification
    class_id, probability = classifier_wrapper(classifier, model_type,
                                               feature_vector)
    return class_id, probability, classes


def file_regression(input_file, model_name, model_type):
    # Load classifier:

    if not os.path.isfile(input_file):
        print("fileClassification: wav file not found!")
        return -1, -1, -1

    regression_models = glob.glob(model_name + "_*")
    regression_models2 = []
    for r in regression_models:
        if r[-5::] != "MEANS":
            regression_models2.append(r)
    regression_models = regression_models2
    regression_names = []
    for r in regression_models:
        regression_names.append(r[r.rfind("_")+1::])

    # FEATURE EXTRACTION
    # LOAD ONLY THE FIRST MODEL (for mt_win, etc)
    if model_type == 'svm' or model_type == "svm_rbf" or \
            model_type == 'randomforest':
        _, _, _, mid_window, mid_step, short_window, short_step, compute_beat \
            = load_model(regression_models[0], True)

    # read audio file and convert to mono
    samping_rate, signal = audioBasicIO.read_audio_file(input_file)
    signal = audioBasicIO.stereo_to_mono(signal)
    # feature extraction:
    mid_features, s, _ = \
        aF.mid_feature_extraction(signal, samping_rate, mid_window * samping_rate,
                                  mid_step * samping_rate,
                                  round(samping_rate * short_window),
                                  round(samping_rate * short_step))
    # long term averaging of mid-term statistics
    mid_features = mid_features.mean(axis=1)
    if compute_beat:
        beat, beat_conf = aF.beat_extraction(s, short_step)
        mid_features = np.append(mid_features, beat)
        mid_features = np.append(mid_features, beat_conf)

    # REGRESSION
    R = []
    for ir, r in enumerate(regression_models):
        if not os.path.isfile(r):
            print("fileClassification: input model_name not found!")
            return (-1, -1, -1)
        if model_type == 'svm' or model_type == "svm_rbf" \
                or model_type == 'randomforest':
            model, mean, std, _, _, _, _, _ = load_model(r, True)
        curFV = (mid_features - mean) / std  # normalization
        R.append(regression_wrapper(model, model_type, curFV))  # classification
    return R, regression_names


def lda(data, labels, red_dim):
    # Centre data
    data -= data.mean(axis=0)
    n_data = np.shape(data)[0]
    n_dim = np.shape(data)[1]
    Sw = np.zeros((n_dim, n_dim))

    C = np.cov((data.T))

    # Loop over classes
    classes = np.unique(labels)
    for i in range(len(classes)):
        # Find relevant datapoints
        indices = (np.where(labels == classes[i]))
        d = np.squeeze(data[indices, :])
        classcov = np.cov((d.T))
        Sw += float(np.shape(indices)[0])/n_data * classcov

    Sb = C - Sw
    # Now solve for W
    # Compute eigenvalues, eigenvectors and sort into order
    evals, evecs = la.eig(Sw, Sb)
    indices = np.argsort(evals)
    indices = indices[::-1]
    evecs = evecs[:, indices]
    w = evecs[:, :red_dim]

    new_data = np.dot(data, w)
    return new_data, w


def write_train_data_arff(model_name, features, classNames, feature_names):
    f = open(model_name + ".arff", 'w')
    f.write('@RELATION ' + model_name + '\n')
    for fn in feature_names:
        f.write('@ATTRIBUTE ' + fn + ' NUMERIC\n')
    f.write('@ATTRIBUTE class {')
    for c in range(len(classNames)-1):
        f.write(classNames[c] + ',')
    f.write(classNames[-1] + '}\n\n')
    f.write('@DATA\n')
    for c, fe in enumerate(features):
        for i in range(fe.shape[0]):
            for j in range(fe.shape[1]):
                f.write("{0:f},".format(fe[i, j]))
            f.write(classNames[c]+"\n")
    f.close()


def train_speaker_models():
    """
    This script is used to train the speaker-related models
    (NOTE: data paths are hard-coded and NOT included in the library,
    the models are, however included)
         import audioTrainTest as aT
        aT.trainSpeakerModelsScript()

    """
    mt_win = 2.0
    mt_step = 2.0
    st_win = 0.020
    st_step = 0.020

    dir_name = "DIARIZATION_ALL/all"
    list_of_dirs = [os.path.join(dir_name, name)
                    for name in os.listdir(dir_name)
                    if os.path.isdir(os.path.join(dir_name, name))]
    extract_features_and_train(list_of_dirs, mt_win, mt_step, st_win, st_step,
                    "knn", "data/knnSpeakerAll",
                               compute_beat=False, train_percentage=0.50)

    dir_name = "DIARIZATION_ALL/female_male"
    list_of_dirs = [os.path.join(dir_name, name)
                    for name in os.listdir(dir_name)
                    if os.path.isdir(os.path.join(dir_name, name))]
    extract_features_and_train(list_of_dirs, mt_win, mt_step, st_win, st_step,
                    "knn", "data/knnSpeakerFemaleMale",
                               compute_beat=False, train_percentage=0.50)


def main(argv):
    return 0


if __name__ == '__main__':
    main(sys.argv)
