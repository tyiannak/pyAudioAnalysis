from __future__ import print_function
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../"))
from pyAudioAnalysis import audioTrainTest as aT
import argparse

def parseArguments():
    parser = argparse.ArgumentParser(prog='PROG')
    parser.add_argument('-d' , '--data_folder',
                        nargs=None,
                        default="/Users/tyiannak/ResearchData/"
                                "Audio Dataset/pyAudioAnalysisData/")
    parser.add_argument('-c', '--classifier_type', nargs=None, required=True,
                        choices = ["knn", "svm", "svm_rbf", "randomforest",
                                   "extratrees", "gradientboosting"],
                        help="Classifier type")
    parser.add_argument('-t', '--task', nargs=None, required=True,
                        choices = ["sm", "movie8", "speakers", "speaker-gender",
                                   "music-genre6", "4class"],
                        help="Classification task")
    args = parser.parse_args()        
    return args



if __name__ == '__main__':
    args = parseArguments()
    root_data_path = args.data_folder
    classifier_type = args.classifier_type

    if args.task == "sm":
        aT.extract_features_and_train([root_data_path +"SM/speech",
                            root_data_path + "SM/music"],
                           1.0, 1.0, 0.05, 0.05, classifier_type,
                           classifier_type + "_sm", False)
    elif args.task == "movie8":
        aT.extract_features_and_train([root_data_path + "movieSegments/8-class/Speech",
                            root_data_path + "movieSegments/8-class/Music",
                            root_data_path + "movieSegments/8-class/Others1",
                            root_data_path + "movieSegments/8-class/Others2",
                            root_data_path + "movieSegments/8-class/Others3",
                            root_data_path + "movieSegments/8-class/Shots",
                            root_data_path + "movieSegments/8-class/Fights",
                            root_data_path + "movieSegments/8-class/Screams"],
                           1.0, 1.0, 0.05, 0.05, classifier_type,
                           classifier_type + "_movie8class", False)
    elif args.task == "speakers":
        aT.extract_features_and_train([root_data_path + "speakerAll/F1",
                            root_data_path + "speakerAll/F2",
                            root_data_path + "speakerAll/F3",
                            root_data_path + "speakerAll/F4",
                            root_data_path + "speakerAll/F5",
                            root_data_path + "speakerAll/M1",
                            root_data_path + "speakerAll/M2",
                            root_data_path + "speakerAll/M3",
                            root_data_path + "speakerAll/M4",
                            root_data_path + "speakerAll/M5"],
                           1.0, 1.0, 0.05, 0.05, classifier_type,
                           classifier_type + "_speaker_10", False)
    elif args.task == "speaker-gender":
        aT.extract_features_and_train([root_data_path + "speakerMaleFemale/Male",
                            root_data_path + "speakerMaleFemale/Female"],
                           1.0, 1.0, 0.05, 0.05, classifier_type,
                           classifier_type + "_speaker_male_female", False)
    elif args.task == "music-genre6":
        aT.extract_features_and_train([root_data_path + "musicalGenreClassification/Blues",
                            root_data_path + "musicalGenreClassification/Classical",
                            root_data_path + "musicalGenreClassification/Electronic",
                            root_data_path + "musicalGenreClassification/Jazz",
                            root_data_path + "musicalGenreClassification/Rap",
                            root_data_path + "musicalGenreClassification/Rock"],
                           1.0, 1.0, 0.05, 0.05, classifier_type,
                          classifier_type + "_musical_genre_6", True)
    elif args.task == "4class":
        aT.extract_features_and_train([root_data_path + "4class/speech",
                            root_data_path + "4class/music",
                            root_data_path + "4class/silence",
                            root_data_path + "4class/other"],
                           1.0, 1.0, 0.05, 0.05, classifier_type,
                           classifier_type + "_4class", False)
