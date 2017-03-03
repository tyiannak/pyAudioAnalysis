import audioBasicIO, sys, csv
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
            xtemp = x[round(T1*Fs):round(T2*Fs)]            
            print T1, T2, label, xtemp.shape
            wavfile.write(label, Fs, xtemp)  

def main(argv):
    wavFile = argv[1]
    annotationFile = argv[2]
    annotation2files(wavFile, annotationFile)

if __name__ == '__main__':
    # This file can be used to extract a series of annotated WAV files based on (a) an audio file (mp3 or wav) and 
    # (b) a segment annotation file e.g. a "label" file generated in audacity
    # usage:
    # python audacityAnnotation2WAVs.py <audiofilepath> <annotationfilepath>
    #
    # The <annotationfilepath> is actually a tab-seperated file where each line has the format <startTime>\t<entTime>\t<classLabel>
    # The result of this process is a  series of WAV files with a file name <audiofilepath>_<startTime>_<endTime>_<classLabel>
    main(sys.argv)
    