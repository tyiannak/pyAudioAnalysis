import glob
import os
import audioBasicIO
import sys
import csv
import scipy.io.wavfile as wavfile


def annotation2files(wavFile, csvFile):
    '''
        Break an audio stream to segments of interest, 
        defined by a csv file
        
        - wavFile:    path to input wavfile
        - csvFile:    path to csvFile of segment limits
        
        Input CSV file must be of the format <T1>\t<T2>\t<Label>
    '''    
    
    [Fs, x] = audioBasicIO.readAudioFile(wavFile)
    with open(csvFile, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for j, row in enumerate(reader):
            T1 = float(row[0].replace(",","."))
            T2 = float(row[1].replace(",","."))            
            label = "%s_%s_%.2f_%.2f.wav" % (wavFile, row[2], T1, T2)
            label = label.replace(" ", "_")
            xtemp = x[int(round(T1*Fs)):int(round(T2*Fs))]            
            print T1, T2, label, xtemp.shape
            wavfile.write(label, Fs, xtemp)  

def main(argv):
    if argv[1] == "-f":
        wavFile = argv[2]
        annotationFile = argv[3]
        annotation2files(wavFile, annotationFile)
    elif argv[1] == "-d":
        inputFolder = argv[2]
        types = ('*.txt', '*.csv')
        annotationFilesList = []
        for files in types:
            annotationFilesList.extend(glob.glob(os.path.join(inputFolder, files)))
        for anFile in annotationFilesList:
            wavFile = os.path.splitext(anFile)[0] + ".wav"
            if not os.path.isfile(wavFile):
                wavFile = os.path.splitext(anFile)[0] + ".mp3"
                if not os.path.isfile(wavFile):
                    print "Audio file not found!"
                    return
            annotation2files(wavFile, anFile)


if __name__ == '__main__':
    # Used to extract a series of annotated WAV files based on (a) an audio file (mp3 or wav) and 
    # (b) a segment annotation file e.g. a "label" file generated in audacity
    #
    # usage 1:
    # python audacityAnnotation2WAVs.py -f <audiofilepath> <annotationfilepath>
    # The <annotationfilepath> is actually a tab-seperated file where each line has the format <startTime>\t<entTime>\t<classLabel>
    # The result of this process is a  series of WAV files with a file name <audiofilepath>_<startTime>_<endTime>_<classLabel>
    # 
    # usage 2:
    # python audacityAnnotation2WAVs.py -d <annotationfolderpath>
    # Same but searches all .txt and .csv annotation files. Audio files are supposed to be in the same path / filename with a WAV extension

    main(sys.argv)
    