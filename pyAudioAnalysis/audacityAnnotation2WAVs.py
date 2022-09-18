import glob
import os
import audioBasicIO
import sys
import csv
import scipy.io.wavfile as wavfile


def annotation2files(wavFile, csvFile):
    """
        Break an audio stream to segments of interest, 
        defined by a csv file
        
        - wavFile:    path to input wavfile
        - csvFile:    path to csvFile of segment limits
        
        Input CSV file must be of the format <T1>\t<T2>\t<Label>
    """
    
    [Fs, x] = audioBasicIO.read_audio_file(wavFile)
    with open(csvFile, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for j, row in enumerate(reader):
            T1 = float(row[0].replace(",","."))
            T2 = float(row[1].replace(",","."))            
            label = "%s_%s_%.2f_%.2f.wav" % (wavFile, row[2], T1, T2)
            label = label.replace(" ", "_")
            xtemp = x[int(round(T1*Fs)):int(round(T2*Fs))]     
            wavfile.write(label, Fs, xtemp)  
            
def annotation2folders(wavFile: str, csvFile: str, folderPath: str):
    """
        Break an audio stream to segments of interest, 
        defined by a csv file
        
        - wavFile:    path to input wavfile
        - csvFile:    path to csvFile of segment limits
        - folderPath: path to class folders
        
        Input CSV file must be of the format <T1>\t<T2>\t<Label>
    """
    [Fs, x] = audioBasicIO.read_audio_file(wavFile)
    with open(csvFile, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for j, row in enumerate(reader):
            T1 = float(row[0].replace(",","."))
            T2 = float(row[1].replace(",","."))
            label = os.path.join(folderPath, row[2].replace(' ', '_'), "%s_%.2f_%.2f.wav" % (os.path.split(wavFile.split)[1], T1, T2))
            if not os.path.exists(os.path.join(folderPath, row[2].replace(' ', '_'))):
                os.makedirs(os.path.join(folderPath, row[2].replace(' ', '_')))
            label = label.replace(" ", "_")
            xtemp = x[int(round(T1*Fs)):int(round(T2*Fs))]            
            wavfile.write(label, Fs, xtemp)
    
def folderAnnotation2folders(sourceFolder, targetFolder):
    """
        Break an audio stream to segments of interest for all files in the sourceFolder
        
        - sourceFolder:    path to Folder of all source audio file and .segments file
        - targetFolder:    path to Folder where user want to store the class folders
    """
    for fileName in glob.glob(os.path.join(sourceFolder, '*.segments')):
        fileName = fileName.split('.')[0]
        annotation2folders('%s.wav' % (fileName), '%s.segments' % (fileName), targetFolder)

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
            annotationFilesList.extend(glob.glob(os.path.join(inputFolder,
                                                              files)))
        for anFile in annotationFilesList:
            wavFile = os.path.splitext(anFile)[0] + ".wav"
            if not os.path.isfile(wavFile):
                wavFile = os.path.splitext(anFile)[0] + ".mp3"
                if not os.path.isfile(wavFile):
                    print("Audio file not found!")
                    return
            annotation2files(wavFile, anFile)


if __name__ == '__main__':
    # Used to extract a series of annotated WAV files based on
    # (a) an audio file (mp3 or wav) and
    # (b) a segment annotation file e.g. a "label" file generated in audacity
    #
    # usage 1:
    # python audacityAnnotation2WAVs.py -f <audiofilepath> <annotationfilepath>
    # The <annotationfilepath> is actually a tab-seperated file where each
    # line has the format <startTime>\t<entTime>\t<classLabel>
    # The result of this process is a  series of WAV files with a file name
    # <audiofilepath>_<startTime>_<endTime>_<classLabel>
    # 
    # usage 2:
    # python audacityAnnotation2WAVs.py -d <annotationfolderpath>
    # Same but searches all .txt and .csv annotation files. Audio files are
    # supposed to be in the same path / filename with a WAV extension

    main(sys.argv)
    