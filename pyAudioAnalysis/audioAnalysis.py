from __future__ import print_function
import argparse
import os
import numpy
import glob
import matplotlib.pyplot as plt
from pyAudioAnalysis import audioFeatureExtraction as aF
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioSegmentation as aS
from pyAudioAnalysis import audioVisualization as aV
from pyAudioAnalysis import audioBasicIO
import scipy.io.wavfile as wavfile
import matplotlib.patches


def dirMp3toWavWrapper(directory, samplerate, channels):
    if not os.path.isdir(directory):
        raise Exception("Input path not found!")

    useMp3TagsAsNames = True
    audioBasicIO.convertDirMP3ToWav(directory, samplerate, channels,
                                    useMp3TagsAsNames)


def dirWAVChangeFs(directory, samplerate, channels):
    if not os.path.isdir(directory):
        raise Exception("Input path not found!")

    audioBasicIO.convertFsDirWavToWav(directory, samplerate, channels)


def featureExtractionFileWrapper(wav_file, out_file, mt_win, mt_step,
                                 st_win, st_step):
    if not os.path.isfile(wav_file):
        raise Exception("Input audio file not found!")

    aF.mtFeatureExtractionToFile(wav_file, mt_win, mt_step, st_win,
                                 st_step, out_file, True, True, True)


def beatExtractionWrapper(wav_file, plot):
    if not os.path.isfile(wav_file):
        raise Exception("Input audio file not found!")
    [fs, x] = audioBasicIO.readAudioFile(wav_file)
    F, _ = aF.stFeatureExtraction(x, fs, 0.050 * fs, 0.050 * fs)
    bpm, ratio = aF.beatExtraction(F, 0.050, plot)
    print("Beat: {0:d} bpm ".format(int(bpm)))
    print("Ratio: {0:.2f} ".format(ratio))


def featureExtractionDirWrapper(directory, mt_win, mt_step, st_win, st_step):
    if not os.path.isdir(directory):
        raise Exception("Input path not found!")
    aF.mtFeatureExtractionToFileDir(directory, mt_win, mt_step, st_win,
                                    st_step, True, True, True)


def featureVisualizationDirWrapper(directory):
    if not os.path.isdir(directory):
        raise Exception("Input folder not found!")
    aV.visualizeFeaturesFolder(directory, "pca", "")
    #aV.visualizeFeaturesFolder(directory, "lda", "artist")


def fileSpectrogramWrapper(wav_file):
    if not os.path.isfile(wav_file):
        raise Exception("Input audio file not found!")
    [fs, x] = audioBasicIO.readAudioFile(wav_file)
    x = audioBasicIO.stereo2mono(x)
    specgram, TimeAxis, FreqAxis = aF.stSpectogram(x, fs, round(fs * 0.040),
                                                   round(fs * 0.040), True)


def fileChromagramWrapper(wav_file):
    if not os.path.isfile(wav_file):
        raise Exception("Input audio file not found!")
    [fs, x] = audioBasicIO.readAudioFile(wav_file)
    x = audioBasicIO.stereo2mono(x)
    specgram, TimeAxis, FreqAxis = aF.stChromagram(x, fs, round(fs * 0.040),
                                                   round(fs * 0.040), True)


def trainClassifierWrapper(method, beat_feats, directories, model_name):
    if len(directories) < 2:
        raise Exception("At least 2 directories are needed")
    aT.featureAndTrain(directories, 1, 1, aT.shortTermWindow, aT.shortTermStep,
                       method.lower(), model_name, compute_beat=beat_feats)


def trainRegressionWrapper(method, beat_feats, dirName, model_name):
    aT.featureAndTrainRegression(dirName, 1, 1, aT.shortTermWindow,
                                 aT.shortTermStep, method.lower(), model_name,
                                 compute_beat=beat_feats)


def classifyFileWrapper(inputFile, model_type, model_name):
    if not os.path.isfile(model_name):
        raise Exception("Input model_name not found!")
    if not os.path.isfile(inputFile):
        raise Exception("Input audio file not found!")

    [Result, P, classNames] = aT.fileClassification(inputFile, model_name,
                                                    model_type)
    print("{0:s}\t{1:s}".format("Class", "Probability"))
    for i, c in enumerate(classNames):
        print("{0:s}\t{1:.2f}".format(c, P[i]))
    print("Winner class: " + classNames[int(Result)])


def regressionFileWrapper(inputFile, model_type, model_name):
    if not os.path.isfile(inputFile):
        raise Exception("Input audio file not found!")

    R, regressionNames = aT.fileRegression(inputFile, model_name, model_type)
    for i in range(len(R)):
        print("{0:s}\t{1:.3f}".format(regressionNames[i], R[i]))


def classifyFolderWrapper(inputFolder, model_type, model_name,
                          outputMode=False):
    if not os.path.isfile(model_name):
        raise Exception("Input model_name not found!")
    types = ('*.wav', '*.aif',  '*.aiff', '*.mp3')
    wavFilesList = []
    for files in types:
        wavFilesList.extend(glob.glob((inputFolder + files)))
    wavFilesList = sorted(wavFilesList)
    if len(wavFilesList) == 0:
        print("No WAV files found!")
        return
    Results = []
    for wavFile in wavFilesList:
        [Result, P, classNames] = aT.fileClassification(wavFile, model_name,
                                                        model_type)
        Result = int(Result)
        Results.append(Result)
        if outputMode:
            print("{0:s}\t{1:s}".format(wavFile, classNames[Result]))
    Results = numpy.array(Results)

    # print distribution of classes:
    [Histogram, _] = numpy.histogram(Results,
                                     bins=numpy.arange(len(classNames) + 1))
    for i, h in enumerate(Histogram):
        print("{0:20s}\t\t{1:d}".format(classNames[i], h))


def regressionFolderWrapper(inputFolder, model_type, model_name):
    files = "*.wav"
    if os.path.isdir(inputFolder):
        strFilePattern = os.path.join(inputFolder, files)
    else:
        strFilePattern = inputFolder + files

    wavFilesList = []
    wavFilesList.extend(glob.glob(strFilePattern))
    wavFilesList = sorted(wavFilesList)
    if len(wavFilesList) == 0:
        print("No WAV files found!")
        return
    Results = []
    for wavFile in wavFilesList:
        R, regressionNames = aT.fileRegression(wavFile, model_name, model_type)
        Results.append(R)
    Results = numpy.array(Results)

    for i, r in enumerate(regressionNames):
        [Histogram, bins] = numpy.histogram(Results[:, i])
        centers = (bins[0:-1] + bins[1::]) / 2.0
        plt.subplot(len(regressionNames), 1, i + 1)
        plt.plot(centers, Histogram)
        plt.title(r)
    plt.show()


def trainHMMsegmenter_fromfile(wavFile, gtFile, hmmModelName, mt_win, mt_step):
    if not os.path.isfile(wavFile):
        print("Error: wavfile does not exist!")
        return
    if not os.path.isfile(gtFile):
        print("Error: groundtruth does not exist!")
        return

    aS.trainHMM_fromFile(wavFile, gtFile, hmmModelName, mt_win, mt_step)


def trainHMMsegmenter_fromdir(directory, hmmModelName, mt_win, mt_step):
    if not os.path.isdir(directory):
        raise Exception("Input folder not found!")
    aS.trainHMM_fromDir(directory, hmmModelName, mt_win, mt_step)


def segmentclassifyFileWrapper(inputWavFile, model_name, model_type):
    if not os.path.isfile(model_name):
        raise Exception("Input model_name not found!")
    if not os.path.isfile(inputWavFile):
        raise Exception("Input audio file not found!")
    gtFile = ""
    if inputWavFile[-4::]==".wav":
        gtFile = inputWavFile.replace(".wav", ".segments")
    if inputWavFile[-4::]==".mp3":
        gtFile = inputWavFile.replace(".mp3", ".segments")
    aS.mtFileClassification(inputWavFile, model_name, model_type, True, gtFile)


def segmentclassifyFileWrapperHMM(wavFile, hmmModelName):
    gtFile = wavFile.replace(".wav", ".segments")
    aS.hmmSegmentation(wavFile, hmmModelName, plot_res=True,
                       gt_file_name=gtFile)


def segmentationEvaluation(dirName, model_name, methodName):
    aS.evaluateSegmentationClassificationDir(dirName, model_name, methodName)


def silenceRemovalWrapper(inputFile, smoothingWindow, weight):
    if not os.path.isfile(inputFile):
        raise Exception("Input audio file not found!")

    [fs, x] = audioBasicIO.readAudioFile(inputFile)
    segmentLimits = aS.silenceRemoval(x, fs, 0.05, 0.05,
                                      smoothingWindow, weight, True)
    for i, s in enumerate(segmentLimits):
        strOut = "{0:s}_{1:.3f}-{2:.3f}.wav".format(inputFile[0:-4], s[0], s[1])
        wavfile.write(strOut, fs, x[int(fs * s[0]):int(fs * s[1])])


def speakerDiarizationWrapper(inputFile, numSpeakers, useLDA):
    if useLDA:
        aS.speakerDiarization(inputFile, numSpeakers, plot_res=True)
    else:
        aS.speakerDiarization(inputFile, numSpeakers, lda_dim=0, plot_res=True)


def thumbnailWrapper(inputFile, thumbnailWrapperSize):
    st_window = 1.0
    st_step = 1.0
    if not os.path.isfile(inputFile):
        raise Exception("Input audio file not found!")

    [fs, x] = audioBasicIO.readAudioFile(inputFile)
    if fs == -1:    # could not read file
        return

    [A1, A2, B1, B2, Smatrix] = aS.musicThumbnailing(x, fs, st_window, st_step,
                                                     thumbnailWrapperSize)

    # write thumbnailWrappers to WAV files:
    if inputFile.endswith(".wav"):
        thumbnailWrapperFileName1 = inputFile.replace(".wav", "_thumb1.wav")
        thumbnailWrapperFileName2 = inputFile.replace(".wav", "_thumb2.wav")
    if inputFile.endswith(".mp3"):
        thumbnailWrapperFileName1 = inputFile.replace(".mp3", "_thumb1.mp3")
        thumbnailWrapperFileName2 = inputFile.replace(".mp3", "_thumb2.mp3")
    wavfile.write(thumbnailWrapperFileName1, fs, x[int(fs * A1):int(fs * A2)])
    wavfile.write(thumbnailWrapperFileName2, fs, x[int(fs * B1):int(fs * B2)])
    print("1st thumbnailWrapper (stored in file {0:s}): {1:4.1f}sec" \
          " -- {2:4.1f}sec".format(thumbnailWrapperFileName1, A1, A2))
    print("2nd thumbnailWrapper (stored in file {0:s}): {1:4.1f}sec" \
          " -- {2:4.1f}sec".format(thumbnailWrapperFileName2, B1, B2))

    # Plot self-similarity matrix:
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect="auto")
    plt.imshow(Smatrix)
    # Plot best-similarity diagonal:
    Xcenter = (A1 / st_step + A2 / st_step) / 2.0
    Ycenter = (B1 / st_step + B2 / st_step) / 2.0

    e1 = matplotlib.patches.Ellipse((Ycenter, Xcenter),
                                    thumbnailWrapperSize * 1.4, 3, angle=45,
                                    linewidth=3, fill=False)
    ax.add_patch(e1)

    plt.plot([B1, Smatrix.shape[0]], [A1, A1], color="k",
             linestyle="--", linewidth=2)
    plt.plot([B2, Smatrix.shape[0]], [A2, A2], color="k",
             linestyle="--", linewidth=2)
    plt.plot([B1, B1], [A1, Smatrix.shape[0]], color="k",
             linestyle="--", linewidth=2)
    plt.plot([B2, B2], [A2, Smatrix.shape[0]], color="k",
             linestyle="--", linewidth=2)

    plt.xlim([0, Smatrix.shape[0]])
    plt.ylim([Smatrix.shape[1], 0])

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    plt.xlabel("frame no")
    plt.ylabel("frame no")
    plt.title("Self-similarity matrix")

    plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser(description="A demonstration script "
                                                 "for pyAudioAnalysis library")
    tasks = parser.add_subparsers(
        title="subcommands", description="available tasks",
        dest="task", metavar="")

    dirMp3Wav = tasks.add_parser("dirMp3toWav",
                                 help="Convert all .mp3 files in a directory "
                                      "to .wav format")
    dirMp3Wav.add_argument("-i", "--input", required=True, help="Input folder")
    dirMp3Wav.add_argument("-r", "--rate", type=int,
                           choices=[8000, 16000, 32000, 44100], required=True,
                           help="Samplerate of generated WAV files")
    dirMp3Wav.add_argument("-c", "--channels", type=int, choices=[1, 2],
                           required=True,
                           help="Audio channels of generated WAV files")

    dirWavRes = tasks.add_parser("dirWavResample",
                                 help="Change samplerate of .wav "
                                      "files in a directory")
    dirWavRes.add_argument("-i", "--input", required=True, help="Input folder")
    dirWavRes.add_argument("-r", "--rate", type=int,
                           choices=[8000, 16000, 32000, 44100], required=True,
                           help="Samplerate of generated WAV files")
    dirWavRes.add_argument("-c", "--channels", type=int, choices=[1, 2],
                           required=True,
                           help="Audio channels of generated WAV files")

    featExt = tasks.add_parser("featureExtractionFile",
                               help="Extract audio features from file")
    featExt.add_argument("-i", "--input", required=True,
                         help="Input audio file")
    featExt.add_argument("-o", "--output", required=True,
                         help="Output file")
    featExt.add_argument("-mw", "--mtwin", type=float,
                         required=True, help="Mid-term window size")
    featExt.add_argument("-ms", "--mtstep", type=float,
                         required=True, help="Mid-term window step")
    featExt.add_argument("-sw", "--stwin", type=float,
                         default=0.050, help="Short-term window size")
    featExt.add_argument("-ss", "--ststep", type=float,
                         default=0.050, help="Short-term window step")

    beat = tasks.add_parser("beatExtraction",
                            help="Compute beat features of an audio file")
    beat.add_argument("-i", "--input", required=True, help="Input audio file")
    beat.add_argument("--plot", action="store_true", help="Generate plot")

    featExtDir = tasks.add_parser("featureExtractionDir",
                                  help="Extract audio features "
                                       "from files in a folder")
    featExtDir.add_argument("-i", "--input", required=True,
                            help="Input directory")
    featExtDir.add_argument("-mw", "--mtwin", type=float, required=True,
                            help="Mid-term window size")
    featExtDir.add_argument("-ms", "--mtstep", type=float, required=True,
                            help="Mid-term window step")
    featExtDir.add_argument("-sw", "--stwin", type=float, default=0.050,
                            help="Short-term window size")
    featExtDir.add_argument("-ss", "--ststep", type=float, default=0.050,
                            help="Short-term window step")

    featVis = tasks.add_parser("featureVisualization")
    featVis.add_argument("-i", "--input", required=True, help="Input directory")

    spectro = tasks.add_parser("fileSpectrogram")
    spectro.add_argument("-i", "--input", required=True,
                         help="Input audio file")

    chroma = tasks.add_parser("fileChromagram")
    chroma.add_argument("-i", "--input", required=True, help="Input audio file")

    trainClass = tasks.add_parser("trainClassifier",
                                  help="Train an SVM or KNN classifier")
    trainClass.add_argument("-i", "--input", nargs="+",
                            required=True, help="Input directories")
    trainClass.add_argument("--method",
                            choices=["svm", "svm_rbf", "knn", "randomforest",
                                     "gradientboosting","extratrees"],
                            required=True, help="Classifier type")
    trainClass.add_argument("--beat", action="store_true",
                            help="Compute beat features")
    trainClass.add_argument("-o", "--output", required=True,
                            help="Generated classifier filename")

    trainReg = tasks.add_parser("trainRegression")
    trainReg.add_argument("-i", "--input", required=True,
                          help="Input directory")
    trainReg.add_argument("--method", choices=["svm", "randomforest","svm_rbf"],
                          required=True, help="Classifier type")
    trainReg.add_argument("--beat", action="store_true",
                          help="Compute beat features")
    trainReg.add_argument("-o", "--output", required=True,
                          help="Generated classifier filename")

    classFile = tasks.add_parser("classifyFile",
                                 help="Classify a file using an "
                                      "existing classifier")
    classFile.add_argument("-i", "--input", required=True,
                           help="Input audio file")
    classFile.add_argument("--model", choices=["svm", "svm_rbf", "knn",
                                               "randomforest",
                                               "gradientboosting",
                                               "extratrees"],
                           required=True, help="Classifier type (svm or knn or"
                                               " randomforest or "
                                               "gradientboosting or "
                                               "extratrees)")
    classFile.add_argument("--classifier", required=True,
                           help="Classifier to use (path)")

    trainHMM = tasks.add_parser("trainHMMsegmenter_fromfile",
                                help="Train an HMM from file + annotation data")
    trainHMM.add_argument("-i", "--input", required=True,
                          help="Input audio file")
    trainHMM.add_argument("--ground", required=True,
                          help="Ground truth path (segments CSV file)")
    trainHMM.add_argument("-o", "--output", required=True,
                          help="HMM model name (path)")
    trainHMM.add_argument("-mw", "--mtwin", type=float, required=True,
                          help="Mid-term window size")
    trainHMM.add_argument("-ms", "--mtstep", type=float, required=True,
                          help="Mid-term window step")

    trainHMMDir = tasks.add_parser("trainHMMsegmenter_fromdir",
                                   help="Train an HMM from file + annotation "
                                        "data stored in a directory (batch)")
    trainHMMDir.add_argument("-i", "--input", required=True,
                             help="Input audio folder")
    trainHMMDir.add_argument("-o", "--output", required=True,
                             help="HMM model name (path)")
    trainHMMDir.add_argument("-mw", "--mtwin", type=float, required=True,
                             help="Mid-term window size")
    trainHMMDir.add_argument("-ms", "--mtstep", type=float, required=True,
                             help="Mid-term window step")

    segmentClassifyFile = tasks.add_parser("segmentClassifyFile",
                                           help="Segmentation - classification "
                                                "of a WAV file given a trained "
                                                "SVM or kNN")
    segmentClassifyFile.add_argument("-i", "--input", required=True,
                                     help="Input audio file")
    segmentClassifyFile.add_argument("--model",
                                     choices=["svm", "svm_rbf", "knn",
                                              "randomforest","gradientboosting",
                                              "extratrees"],
                                     required=True, help="Model type")
    segmentClassifyFile.add_argument("--modelName", required=True,
                                     help="Model path")

    segmentClassifyFileHMM = tasks.add_parser("segmentClassifyFileHMM",
                                              help="Segmentation - "
                                                   "classification of a WAV "
                                                   "file given a trained HMM")
    segmentClassifyFileHMM.add_argument("-i", "--input", required=True,
                                        help="Input audio file")
    segmentClassifyFileHMM.add_argument("--hmm", required=True,
                                        help="HMM Model to use (path)")

    segmentationEvaluation = tasks.add_parser("segmentationEvaluation", help=
                                              "Segmentation - classification "
                                              "evaluation for a list of WAV "
                                              "files and CSV ground-truth "
                                              "stored in a folder")
    segmentationEvaluation.add_argument("-i", "--input", required=True,
                                        help="Input audio folder")
    segmentationEvaluation.add_argument("--model",
                                        choices=["svm", "knn", "hmm"],
                                        required=True, help="Model type")
    segmentationEvaluation.add_argument("--modelName", required=True,
                                        help="Model path")

    regFile = tasks.add_parser("regressionFile")
    regFile.add_argument("-i", "--input", required=True,
                         help="Input audio file")
    regFile.add_argument("--model", choices=["svm", "svm_rbf","randomforest"],
                         required=True, help="Regression type")
    regFile.add_argument("--regression", required=True,
                         help="Regression model to use")

    classFolder = tasks.add_parser("classifyFolder")
    classFolder.add_argument("-i", "--input", required=True,
                             help="Input folder")
    classFolder.add_argument("--model", choices=["svm", "svm_rbf", "knn",
                                                 "randomforest",
                                                 "gradientboosting",
                                                 "extratrees"],
                             required=True, help="Classifier type")
    classFolder.add_argument("--classifier", required=True,
                             help="Classifier to use (filename)")
    classFolder.add_argument("--details", action="store_true",
                             help="Plot details (otherwise only "
                                  "counts per class are shown)")

    regFolder = tasks.add_parser("regressionFolder")
    regFolder.add_argument("-i", "--input", required=True, help="Input folder")
    regFolder.add_argument("--model", choices=["svm", "knn"],
                           required=True, help="Classifier type")
    regFolder.add_argument("--regression", required=True,
                           help="Regression model to use")

    silrem = tasks.add_parser("silenceRemoval",
                              help="Remove silence segments from a recording")
    silrem.add_argument("-i", "--input", required=True, help="input audio file")
    silrem.add_argument("-s", "--smoothing", type=float, default=1.0,
                        help="smoothing window size in seconds.")
    silrem.add_argument("-w", "--weight", type=float, default=0.5,
                        help="weight factor in (0, 1)")

    spkrDir = tasks.add_parser("speakerDiarization")
    spkrDir.add_argument("-i", "--input", required=True,
                         help="Input audio file")
    spkrDir.add_argument("-n", "--num", type=int, required=True,
                         help="Number of speakers")
    spkrDir.add_argument("--flsd", action="store_true",
                         help="Enable FLsD method")

    speakerDiarizationScriptEval = tasks.add_parser("speakerDiarizationScriptEval",
                                                    help="Train an SVM or KNN "
                                                         "classifier")
    speakerDiarizationScriptEval.add_argument("-i", "--input", required=True,
                                              help="Input directory")
    speakerDiarizationScriptEval.add_argument("--LDAs", type=int, nargs="+",
                                              required=True,
                                              help="List FLsD params")

    thumb = tasks.add_parser("thumbnail",
                             help="Generate a thumbnailWrapper "
                                  "for an audio file")
    thumb.add_argument("-i", "--input", required=True, help="input audio file")
    thumb.add_argument("-s", "--size",  default=10.0,  type=float,
                       help="thumbnailWrapper size in seconds.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()


    if args.task == "dirMp3toWav":
        # Convert mp3 to wav (batch - folder)
        dirMp3toWavWrapper(args.input, args.rate, args.channels)
    elif args.task == "dirWavResample":
        # Convert fs for a list of wavs stored in a folder
        dirWAVChangeFs(args.input, args.rate, args.channels)
    elif args.task == "featureExtractionFile":
        # Feature extraction for WAV file
        featureExtractionFileWrapper(args.input, args.output, args.mtwin,
                                     args.mtstep, args.stwin, args.ststep)
    elif args.task == "featureExtractionDir":
        # Feature extraction for all WAV files stored in a folder
        featureExtractionDirWrapper(args.input, args.mtwin, args.mtstep,
                                    args.stwin, args.ststep)
    elif args.task == "fileSpectrogram":
        # Extract spectrogram from a WAV file
        fileSpectrogramWrapper(args.input)
    elif args.task == "fileChromagram":
        # Extract chromagram from a WAV file
        fileChromagramWrapper(args.input)
    elif args.task == "featureVisualization":
        # Visualize the content of a list of WAV files stored in a folder
        featureVisualizationDirWrapper(args.input)
    elif args.task == "beatExtraction":
        # Extract bpm from file
        beatExtractionWrapper(args.input, args.plot)
    elif args.task == "trainClassifier":
        # Train classifier from data (organized in folders)
        trainClassifierWrapper(args.method, args.beat, args.input, args.output)
    elif args.task == "trainRegression":
        # Train a regression model from data (organized in
        # a single folder, while ground-truth is provided in a CSV)
        trainRegressionWrapper(args.method, args.beat, args.input, args.output)
    elif args.task == "classifyFile":
        # Apply audio classifier on audio file
        classifyFileWrapper(args.input, args.model, args.classifier)
    elif args.task == "trainHMMsegmenter_fromfile":
        # Train an hmm segmenter-classifier from WAV file + annotation
        trainHMMsegmenter_fromfile(args.input, args.ground, args.output,
                                   args.mtwin, args.mtstep)
    elif args.task == "trainHMMsegmenter_fromdir":
        # Train an hmm segmenter-classifier from a list of
        # WAVs and annotations stored in a folder
        trainHMMsegmenter_fromdir(args.input, args.output, args.mtwin,
                                  args.mtstep)
    elif args.task == "segmentClassifyFile":
        # Apply a classifier (svm or knn or randomforest or gradientboosting
        # or extratrees) for segmentation-classificaiton to a WAV file
        segmentclassifyFileWrapper(args.input, args.modelName, args.model)
    elif args.task == "segmentClassifyFileHMM":
        # Apply an hmm for segmentation-classificaiton to a WAV file
        segmentclassifyFileWrapperHMM(args.input, args.hmm)
    elif args.task == "segmentationEvaluation":
        # Evaluate segmentation-classification for a list of WAV files
        # (and ground truth CSVs) stored in a folder
        segmentationEvaluation(args.input, args.modelName, args.model)
    elif args.task == "regressionFile":
        # Apply a regression model to an audio signal stored in a WAV file
        regressionFileWrapper(args.input, args.model, args.regression)
    elif args.task == "classifyFolder":
        # Classify every WAV file in a given path
        classifyFolderWrapper(args.input, args.model, args.classifier,
                              args.details)
    elif args.task == "regressionFolder":
        # Apply a regression model on every WAV file in a given path
        regressionFolderWrapper(args.input, args.model, args.regression)
    elif args.task == "silenceRemoval":
        # Detect non-silent segments in a WAV file and
        # output to seperate WAV files
        silenceRemovalWrapper(args.input, args.smoothing, args.weight)
    elif args.task == "speakerDiarization":
        # Perform speaker diarization on a WAV file
        speakerDiarizationWrapper(args.input, args.num, args.flsd)
    elif args.task == "speakerDiarizationScriptEval":
        # Evaluate speaker diarization given a folder that contains
        # WAV files and .segment (Groundtruth files)
        aS.speakerDiarizationEvaluateScript(args.input, args.LDAs)
    elif args.task == "thumbnail":
        # Audio thumbnailing
        thumbnailWrapper(args.input, args.size)
