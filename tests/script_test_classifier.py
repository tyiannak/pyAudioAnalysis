from __future__ import print_function
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioSegmentation as aS
import matplotlib.pyplot as plt
import argparse

def parseArguments():
    parser = argparse.ArgumentParser(prog='PROG')
    parser.add_argument('-d' , '--data_folder', nargs=None, default="/Users/tyiannak/ResearchData/Audio Dataset/pyAudioAnalysisData/")
    parser.add_argument('-c' , '--classifier_type', nargs=None, required=True, 
                        choices = ["knn", "svm", "svm_rbf", "randomforest", "extratrees", "gradientboosting"],
                        help="Classifier type")
    args = parser.parse_args()        
    return args



if __name__ == '__main__':
    args = parseArguments()
    root_data_path = args.data_folder
    classifier_type = args.classifier_type
    classifier_path = "sm_" + classifier_type
    aT.featureAndTrain([root_data_path +"SM/speech",root_data_path + "SM/music"], 
                       1.0, 1.0, 0.2, 0.2, classifier_type, 
                       classifier_path, False)
