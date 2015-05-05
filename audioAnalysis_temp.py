#!/usr/bin/env python2.7

import argparse
import os, audioop, numpy, glob,  scipy, subprocess, wave, cPickle, threading, shutil, ntpath
import matplotlib.pyplot as plt
import audioFeatureExtraction as aF    
import audioTrainTest as aT
import audioSegmentation as aS
import audioVisualization as aV
import audioBasicIO
import utilities as uT
import scipy.io.wavfile as wavfile
import matplotlib.patches


def dirMp3toWav(directory, samplerate, channels):
    if not os.path.isdir(directory):
        raise Exception("Input path not found!")
        
    useMp3TagsAsNames = True
    audioBasicIO.convertDirMP3ToWav(directory, samplerate, channels, useMp3TagsAsNames)
    
def dirWAVChangeFs(directory, samplerate, channels):
    if not os.path.isdir(directory):
        raise Exception("Input path not found!")
        
    audioBasicIO.convertFsDirWavToWav(directory, samplerate, channels)
    
def featureExtractionFile(wavFileName, outFile, mtWin, mtStep, stWin, stStep):
    if not os.path.isfile(wavFileName):
        raise Exception("Input audio file not found!")
    
    aF.mtFeatureExtractionToFile(wavFileName, mtWin, mtStep, stWin, stStep, outFile, True, True, True)
    
def beatExtraction(wavFileName, plot):
    if not os.path.isfile(wavFileName):
        raise Exception("Input audio file not found!")
    
    [Fs, x] = audioBasicIO.readAudioFile(wavFileName);
    F = aF.stFeatureExtraction(x, Fs, 0.050*Fs, 0.050*Fs);
    BPM, ratio = aF.beatExtraction(F, 0.050, plot)
    print "Beat: {0:d} bpm ".format(int(BPM))
    print "Ratio: {0:.2f} ".format(ratio)
        
def featureExtractionDir(directory, mtWin, mtStep, stWin, stStep):
    if not os.path.isdir(directory):
        raise Exception("Input path not found!")
    
    aF.mtFeatureExtractionToFileDir(directory, mtWin, mtStep, stWin, stStep, True, True, True)

def featureVisualizationDir(directory):
    if not os.path.isdir(directory):
        raise Exception("Input folder not found!")
    
    aV.visualizeFeaturesFolder(directory, "pca", "")

def fileSpectrogram(wavFileName):
    if not os.path.isfile(wavFileName):
        raise Exception("Input audio file not found!")
    
    [Fs, x] = audioBasicIO.readAudioFile(wavFileName)
    x = audioBasicIO.stereo2mono(x)
    specgram, TimeAxis, FreqAxis = aF.stSpectogram(x, Fs, round(Fs*0.040), round(Fs*0.040), True)

def fileChromagram(wavFileName):
    if not os.path.isfile(wavFileName):
        raise Exception("Input audio file not found!")
    
    [Fs, x] = audioBasicIO.readAudioFile(wavFileName)
    x = audioBasicIO.stereo2mono(x)
    specgram, TimeAxis, FreqAxis = aF.stChromagram(x, Fs, round(Fs*0.040), round(Fs*0.040), True)

def trainClassifier(method, beatFeatures, directories, modelName):
    if len(directories) < 2:
        raise Exception("At least 2 directories are needed")
    
    aT.featureAndTrain(directories, 1, 1, aT.shortTermWindow, aT.shortTermStep, 
        method.lower(), modelName, computeBEAT = beatFeatures)
    
def trainRegression(method, beatFeatures, directories, modelName):
    if len(directories) < 2:
        raise Exception("At least 2 directories are needed")
    
    aT.featureAndTrainRegression(dirName, 1, 1, aT.shortTermWindow, aT.shortTermStep, 
        method.lower(), modelName, computeBEAT = beatFeatures)
        
def classifyFile(inputFile, modelType, modelName):
    if not os.path.isfile(modelName):
        raise Exception("Input modelName not found!")
    if not os.path.isfile(inputFile):
        raise Exception("Input audio file not found!")
        
    [Result, P, classNames] = aT.fileClassification(inputFile, modelName, modelType)
    print "{0:s}\t{1:s}".format("Class","Probability")
    for i,c in enumerate(classNames):
        print "{0:s}\t{1:.2f}".format(c,P[i])
    print "Winner class: " + classNames[int(Result)]
    
def regressionFile(inputFile, modelType, modelName):
    if not os.path.isfile(modelName):
        raise Exception("Input modelName not found!")
    if not os.path.isfile(inputFile):
        raise Exception("Input audio file not found!")
        
    R, regressionNames = aT.fileRegression(inputFile, modelName, modelType)
    for i in range(len(R)):
        print "{0:s}\t{1:.3f}".format(regressionNames[i], R[i])

def classifyFolder(inputFolder, modelType, modelName, outputMode=False):
    if not os.path.isfile(modelName):
        raise Exception("Input modelName not found!")
        
    files = "*.wav"
    if os.path.isdir(inputFolder):
        strFilePattern = os.path.join(inputFolder, files)
    else:
        strFilePattern = inputFolder + files

    wavFilesList = []
    wavFilesList.extend(glob.glob(strFilePattern))
    wavFilesList = sorted(wavFilesList)
    if len(wavFilesList)==0:
        print "No WAV files found!"
        return 
    
    Results = []
    for wavFile in wavFilesList:    
        [Result, P, classNames] = aT.fileClassification(wavFile, modelName, modelType)    
        Result = int(Result)
        Results.append(Result)
        if outputMode:
            print "{0:s}\t{1:s}".format(wavFile,classNames[Result])
    Results = numpy.array(Results)
    
    # print distribution of classes:
    [Histogram, _] = numpy.histogram(Results, bins=numpy.arange(len(classNames)+1))
    for i,h in enumerate(Histogram):
        print "{0:20s}\t\t{1:d}".format(classNames[i], h)
    
def regressionFolder(inputFolder, modelType, modelName):
    files = "*.wav"
    if os.path.isdir(inputFolder):
        strFilePattern = os.path.join(inputFolder, files)
    else:
        strFilePattern = inputFolder + files

    wavFilesList = []
    wavFilesList.extend(glob.glob(strFilePattern))
    wavFilesList = sorted(wavFilesList)    
    if len(wavFilesList)==0:
        print "No WAV files found!"
        return 
    Results = []
    for wavFile in wavFilesList:    
        R, regressionNames = aT.fileRegression(wavFile, modelName, modelType)
        Results.append(R)
    Results = numpy.array(Results)
    
    for i, r in enumerate(regressionNames):
        [Histogram, bins] = numpy.histogram(Results[:, i])
        centers = (bins[0:-1] + bins[1::]) / 2.0
        plt.subplot(len(regressionNames), 1, i);
        plt.plot(centers, Histogram)
        plt.title(r)
    plt.show()

def trainHMMsegmenter_fromfile(wavFile, gtFile, hmmModelName, mtWin, mtStep):
    if not os.path.isfile(wavFile):
        print "Error: wavfile does not exist!"; return
    if not os.path.isfile(gtFile):
        print "Error: groundtruth does not exist!"; return
        
    aS.trainHMM_fromFile(wavFile, gtFile, hmmModelName, mtWin, mtStep)
    
def trainHMMsegmenter_fromdir(directory, hmmModelName, mtWin, mtStep):
    if not os.path.isdir(directory):
        raise Exception("Input folder not found!")
        
    aS.trainHMM_fromDir(directory, hmmModelName, mtWin, mtStep)
    
def segmentClassifyFileHMM(wavFile, hmmModelName):
    gtFile = wavFile.replace(".wav", ".segments");            
    aS.hmmSegmentation(wavFile, hmmModelName, PLOT = True, gtFileName = gtFile)
    
def segmentClassifyFile(inputWavFile, modelName, modelType):
    if not os.path.isfile(modelName):
        raise Exception("Input modelName not found!")
    if not os.path.isfile(inputWavFile):
        raise Exception("Input audio file not found!")
    
    gtFile = inputWavFile.replace(".wav", ".segments")
    aS.mtFileClassification(inputWavFile, modelName, modelType, True, gtFile)

def segmentationEvaluation(dirName, modelName, methodName):
    aS.evaluateSegmentationClassificationDir(dirName, modelName, methodName)

def silenceRemoval(inputFile, smoothingWindow, weight):
    if not os.path.isfile(inputFile):
        raise Exception("Input audio file not found!")
    
    [Fs, x] = audioBasicIO.readAudioFile(inputFile)                        # read audio signal
    segmentLimits = aS.silenceRemoval(x, Fs, 0.05, 0.05, smoothingWindow, weight, True)    # get onsets
    for i, s in enumerate(segmentLimits):
        strOut = "{0:s}_{1:.3f}-{2:.3f}.wav".format(inputFile[0:-4], s[0], s[1])
        wavfile.write( strOut, Fs, x[int(Fs*s[0]):int(Fs*s[1])])
        
def speakerDiarization(inputFile, numSpeakers):
    aS.speakerDiarization(inputFile, 2.0, 0.1, numSpeakers)

def thumbnail(inputFile, thumbnailSize):
    stWindow = 1.0
    stStep = 1.0
    if not os.path.isfile(inputFile):
        raise Exception("Input audio file not found!")

    [Fs, x] = audioBasicIO.readAudioFile(inputFile)                        # read file
    if Fs == -1:    # could not read file
        return

    [A1, A2, B1, B2, Smatrix] = aS.musicThumbnailing(x, Fs, stWindow, stStep, thumbnailSize)    # find thumbnail endpoints            

    # write thumbnails to WAV files:
    thumbnailFileName1 = inputFile.replace(".wav","_thumb1.wav")
    thumbnailFileName2 = inputFile.replace(".wav","_thumb2.wav")
    wavfile.write(thumbnailFileName1, Fs, x[int(Fs*A1):int(Fs*A2)])
    wavfile.write(thumbnailFileName2, Fs, x[int(Fs*B1):int(Fs*B2)])
    print "1st thumbnail (stored in file {0:s}): {1:4.1f}sec -- {2:4.1f}sec".format(thumbnailFileName1, A1, A2)
    print "2nd thumbnail (stored in file {0:s}): {1:4.1f}sec -- {2:4.1f}sec".format(thumbnailFileName2, B1, B2)

    # Plot self-similarity matrix:
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect="auto")
    plt.imshow(Smatrix)
    # Plot best-similarity diagonal:
    Xcenter = (A1/stStep + A2/stStep) / 2.0
    Ycenter = (B1/stStep + B2/stStep) / 2.0

    e1 = matplotlib.patches.Ellipse((Ycenter, Xcenter), thumbnailSize * 1.4, 3,
             angle=45, linewidth=3, fill=False)
    ax.add_patch(e1)

    plt.plot([B1, Smatrix.shape[0]], [A1, A1], color="k", linestyle="--", linewidth=2)
    plt.plot([B2, Smatrix.shape[0]], [A2, A2], color="k", linestyle="--", linewidth=2)
    plt.plot([B1, B1], [A1, Smatrix.shape[0]], color="k", linestyle="--", linewidth=2)
    plt.plot([B2, B2], [A2, Smatrix.shape[0]], color="k", linestyle="--", linewidth=2)

    plt.xlim([0, Smatrix.shape[0]])
    plt.ylim([Smatrix.shape[1], 0])

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    plt.xlabel("frame no")
    plt.ylabel("frame no")
    plt.title("Self-similarity matrix")

    plt.show()
        
def parse_arguments():
    parser = argparse.ArgumentParser(description="A demonstration script for pyAudioAnalysis library")
    tasks = parser.add_subparsers(
        title="subcommands", description="available tasks", dest="task", metavar="")
    
    dirMp3Wav = tasks.add_parser("dirMp3toWav", help="Convert .mp3 files in a directory to .wav format")
    dirMp3Wav.add_argument("-i", "--input", required=True, help="Input folder")
    dirMp3Wav.add_argument("-r", "--rate",  type=int, choices=[8000, 16000, 32000, 44100], 
            required=True, help="Samplerate of generated WAV files")
    dirMp3Wav.add_argument("-c", "--channels", type=int, choices=[1, 2], 
            required=True, help="Audio channels of generated WAV files")
    
    dirWavRes = tasks.add_parser("dirWavResample", help="Change samplerate of .wav files in a directory")
    dirWavRes.add_argument("-i", "--input", required=True, help="Input folder")
    dirWavRes.add_argument("-r", "--rate",  type=int, choices=[8000, 16000, 32000, 44100], 
            required=True, help="Samplerate of generated WAV files")
    dirWavRes.add_argument("-c", "--channels", type=int, choices=[1, 2], 
            required=True, help="Audio channels of generated WAV files")
    
    featExt = tasks.add_parser("featureExtraction", help="Extract audio features from file")
    featExt.add_argument("-i", "--input",               required=True, help="Input audio file")
    featExt.add_argument("-o", "--output",              required=True, help="Output file")
    featExt.add_argument("-mw", "--mtwin",  type=float, required=True, help="Mid-term window size")
    featExt.add_argument("-ms", "--mtstep", type=float, required=True, help="Mid-term window step")
    featExt.add_argument("-sw", "--stwin",  type=float, required=True, help="Short-term window size")
    featExt.add_argument("-ss", "-ststep",  type=float, required=True, help="Short-term window step")

    beat = tasks.add_parser("beatExtraction", help="Compute beat features of an audio file")
    beat.add_argument("-i", "--input", required=True, help="Input audio file")
    beat.add_argument("--plot", action="store_true",  help="Generate plot")
    
    featExtDir = tasks.add_parser("featureExtractionDir", help="Extract audio features from files in a folder")
    featExtDir.add_argument("-i", "--input",               required=True, help="Input directory")
    featExtDir.add_argument("-mw", "--mtwin",  type=float, required=True, help="Mid-term window size")
    featExtDir.add_argument("-ms", "--mtstep", type=float, required=True, help="Mid-term window step")
    featExtDir.add_argument("-sw", "--stwin",  type=float, required=True, help="Short-term window size")
    featExtDir.add_argument("-ss", "-ststep",  type=float, required=True, help="Short-term window step")

    featVis = tasks.add_parser("featureVisualization")
    featVis.add_argument("-i", "--input", required=True, help="Input directory")
    
    spectro = tasks.add_parser("spectrogram")
    spectro.add_argument("-i", "--input", required=True, help="Input audio file")
    
    chroma = tasks.add_parser("chromagram")
    chroma.add_argument("-i", "--input", required=True, help="Input audio file")
    
    trainClass = tasks.add_parser("trainClassifier", help="Train an SVM or KNN classifier")
    trainClass.add_argument("-i", "--input", nargs="+",         required=True, help="Input directories")
    trainClass.add_argument("--method", choices=["svm", "knn"], required=True, help="Classifier type")
    trainClass.add_argument("--beat", action="store_true",   help="Compute beat features")
    trainClass.add_argument("-o", "--output", required=True, help="Generated classifier filename")
    
    trainReg = tasks.add_parser("trainRegression")
    trainReg.add_argument("-i", "--input", required=True, nargs="+", help="Input directories")
    trainReg.add_argument("--method", choices=["svm", "knn"], required=True, help="Classifier type")
    trainReg.add_argument("--beat", action="store_true",   help="Compute beat features")
    trainReg.add_argument("-o", "--output", required=True, help="Generated classifier filename")
    
    classFile = tasks.add_parser("classifyFile", help="Classify a file using an existing classifier")
    classFile.add_argument("-i", "--input",                   required=True, help="Input audio file")
    classFile.add_argument("--model", choices=["svm", "knn"], required=True, help="Classifier type")
    classFile.add_argument("--classifier",                    required=True, help="Classifier to use")
    
    regFile = tasks.add_parser("regressionFile")
    regFile.add_argument("-i", "--input",                   required=True, help="Input audio file")
    regFile.add_argument("--model", choices=["svm", "knn"], required=True, help="Regression type")
    regFile.add_argument("--regression",                    required=True, help="Regression model to use")
    
    classFolder = tasks.add_parser("classifyFolder")
    classFolder.add_argument("-i", "--input",                   required=True, help="Input folder")
    classFolder.add_argument("--model", choices=["svm", "knn"], required=True, help="Classifier type")
    classFolder.add_argument("--classifier",                    required=True, help="Classifier to use")
    
    regFolder = tasks.add_parser("regressionFolder")
    regFolder.add_argument("-i", "--input",                   required=True, help="Input folder")
    regFolder.add_argument("--model", choices=["svm", "knn"], required=True, help="Classifier type")
    regFolder.add_argument("--regression",                    required=True, help="Regression model to use")

    silrem = tasks.add_parser("silenceRemoval", help="Remove silence segments from a recording")
    silrem.add_argument("-i", "--input", required=True,               help="input audio file")
    silrem.add_argument("-s", "--smoothing", type=float, default=0.5, help="smoothing window size in seconds.")
    silrem.add_argument("-w", "--weight",    type=float, default=0.5, help="weight factor in (0, 1)")
    
    spkrDir = tasks.add_parser("speakerDiarization")
    spkrDir.add_argument("-i", "--input",         required=True, help="Input audio file")
    spkrDir.add_argument("-n", "--num", type=int, required=True, help="Number of speakers")
    
    thumb = tasks.add_parser("thumbnail", help="Generate a thumbnail for an audio file")
    thumb.add_argument("-i", "--input", required=True, help="input audio file")
    thumb.add_argument("-s", "--size",  default=10.0,  help="thumbnail size in seconds.")
    
    return parser.parse_args()
        
    
if __name__ == "__main__":
    args = parse_arguments()

    if args.task == "dirMp3ToWav":
        dirMp3toWav(args.input, args.rate, args.channels)
    elif args.task == "dirWavResample":
        dirWAVChangeFs(args.input, args.rate, args.channels)
    elif args.task == "featureExtraction":
        featureExtractionFile(args.input, args.output, 
                args.mtwin, args.mtstep, args.stwin, args.ststep)
    elif args.task == "beatExtraction":
        beatExtraction(args.input, args.plot)
    elif args.task == "featureExtractionDir":
        featureExtractionDir(args.input, args.mtwin, args.mtstep, args.stwin, args.ststep)
    elif args.task == "featureVisualization":
        featureVisualizationDir(args.input)
    elif args.task == "spectrogram":
        fileSpectrogram(args.input)
    elif args.task == "chromagram":
        fileChromagram(args.input)
    elif args.task == "trainClassifier":
        trainClassifier(args.method, args.beat, args.input, args.output)
    elif args.task == "trainRegression":
        trainRegression(args.method, args.beat, args.input, args.output)
    elif args.task == "classifyFile":
        classifyFile(args.input, args.model, args.classifier)
    elif args.task == "regressionFile":
        regressionFile(args.input, args.model, args.regression)
    elif args.task == "classifyFolder":
        classifyFolder(args.input, args.model, args.classifier)
    elif args.task == "regressionFolder":
        regressionFolder(args.input, args.model, args.regression)
    elif args.task == "silenceRemoval":
        silenceRemoval(args.input, args.smoothing, args.weight)
    elif args.task == "speakerDiarization":
        speakerDiarization(args.input, args.num)
    elif args.task == "thumbnail":
        thumbnail(args.input, args.size)
