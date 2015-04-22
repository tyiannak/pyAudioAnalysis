import sys
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioSegmentation as aS
import matplotlib.pyplot as plt
import time

def main(argv):
	if argv[1] == "-shortTerm":
		[Fs, x] = audioBasicIO.readAudioFile("diarizationExample.wav");
		duration = x.shape[0] / float(Fs)
		t1 = time.clock()
		F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.050*Fs);
		t2 = time.clock()
		perTime1 =  duration / (t2-t1); print "short-term feature extraction: {0:.1f} x realtime".format(perTime1)
	elif argv[1] == "-classifyFile":
		[Fs, x] = audioBasicIO.readAudioFile("diarizationExample.wav");
		duration = x.shape[0] / float(Fs)		
		t1 = time.clock()
		aT.fileClassification("diarizationExample.wav", "svmSM","svm")
		t2 = time.clock()
		perTime1 =  duration / (t2-t1); print "Mid-term feature extraction + classification: {0:.1f} x realtime".format(perTime1)
	elif argv[1] == "-mtClassify":
		[Fs, x] = audioBasicIO.readAudioFile("diarizationExample.wav");
		duration = x.shape[0] / float(Fs)		
		t1 = time.clock()
		[flagsInd, classesAll, acc] = aS.mtFileClassification("diarizationExample.wav", "svmSM", "svm", False, '')
		t2 = time.clock()
		perTime1 =  duration / (t2-t1); print "Fix-sized classification - segmentation: {0:.1f} x realtime".format(perTime1)
	elif argv[1] == "-hmmSegmentation":
		[Fs, x] = audioBasicIO.readAudioFile("diarizationExample.wav");
		duration = x.shape[0] / float(Fs)		
		t1 = time.clock()
		aS.hmmSegmentation('diarizationExample.wav', 'hmmRadioSM', False, '')             
		t2 = time.clock()
		perTime1 =  duration / (t2-t1); print "HMM-based classification - segmentation: {0:.1f} x realtime".format(perTime1)
	elif argv[1] == "-silenceRemoval":
		[Fs, x] = audioBasicIO.readAudioFile("diarizationExample.wav");
		duration = x.shape[0] / float(Fs)				
		t1 = time.clock()
		[Fs, x] = audioBasicIO.readAudioFile("diarizationExample.wav");
		segments = aS.silenceRemoval(x, Fs, 0.050, 0.050, smoothWindow = 1.0, Weight = 0.3, plot = False)
		t2 = time.clock()
		perTime1 =  duration / (t2-t1); print "Silence removal: {0:.1f} x realtime".format(perTime1)
	elif argv[1] == "-thumbnailing":
		[Fs1, x1] = audioBasicIO.readAudioFile("scottish.wav")
		duration1 = x1.shape[0] / float(Fs1)		
		t1 = time.clock()
		[A1, A2, B1, B2, Smatrix] = aS.musicThumbnailing(x1, Fs1, 1.0, 1.0, 15.0)	# find thumbnail endpoints			
		t2 = time.clock()
		perTime1 =  duration1 / (t2-t1); print "Thumbnail: {0:.1f} x realtime".format(perTime1)
	elif argv[1] == "-diarization-noLDA":
		[Fs1, x1] = audioBasicIO.readAudioFile("../DiarizationDataStreams/07-02-28_manual.wav")
		duration1 = x1.shape[0] / float(Fs1)		
		t1 = time.clock()		
		aS.speakerDiarization("../DiarizationDataStreams/07-02-28_manual.wav", 4, LDAdim = 0, PLOT = False)
		t2 = time.clock()
		perTime1 =  duration1 / (t2-t1); print "Diarization: {0:.1f} x realtime".format(perTime1)
	elif argv[1] == "-diarization-LDA":
		[Fs1, x1] = audioBasicIO.readAudioFile("../DiarizationDataStreams/07-02-28_manual.wav")
		duration1 = x1.shape[0] / float(Fs1)		
		t1 = time.clock()		
		aS.speakerDiarization("../DiarizationDataStreams/07-02-28_manual.wav", 4, PLOT = False)
		t2 = time.clock()
		perTime1 =  duration1 / (t2-t1); print "Diarization: {0:.1f} x realtime".format(perTime1)
		
if __name__ == '__main__':
	main(sys.argv)



