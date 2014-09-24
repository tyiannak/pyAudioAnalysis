import os, glob
import scipy.io.wavfile as wavfile

def convertDirMP3ToWav(dirName, Fs, nC):
	types = (dirName+os.sep+'*.mp3',) # the tuple of file types
	filesToProcess = []
	for files in types:
		filesToProcess.extend(glob.glob(files))		
	for f in filesToProcess:
		command = "avconv -i \"" + f + "\" -ar " +str(Fs) + " -ac " + str(nC) + " \"" + f.replace(".mp3",".wav") + "\"";
		print command
		os.system(command)

def readAudioFile(path):
	extension = os.path.splitext(path)[1]

	try:
		if extension.lower() == '.wav':
			[Fs, x] = wavfile.read(path)
		elif extension.lower() == '.aif' or extension.lower() == '.aiff':
			s = aifc.open(path, 'r')
			nframes = s.getnframes()
			strsig = s.readframes(nframes)
			x = numpy.fromstring(strsig, numpy.short).byteswap()
			Fs = s.getframerate()
		else:
			print "Error in readAudioFile(): Unknown file type!"
			return (-1,-1)
	except IOError:	
		print "Error: file not found or other I/O error."
		return (-1,-1)
	return (Fs, x)

def stereo2mono(x):
	if x.ndim==1:
		return x
	else:
		if x.ndim==2:
			return ( (x[:,1] / 2) + (x[:,0] / 2) )
		else:
			return -1

