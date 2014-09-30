import os, glob, eyeD3, ntpath
import scipy.io.wavfile as wavfile

def convertDirMP3ToWav(dirName, Fs, nC, useMp3TagsAsName = False):
	'''
	This function converts the MP3 files stored in a folder to WAV. If required, the output names of the WAV files are based on MP3 tags, otherwise the same names are used.
	ARGUMENTS:
	 - dirName:		the path of the folder where the MP3s are stored
	 - Fs:			the sampling rate of the generated WAV files
	 - nC:			the number of channesl of the generated WAV files
	 - useMp3TagsAsName: 	True if the WAV filename is generated on MP3 tags
	'''

	types = (dirName+os.sep+'*.mp3',) # the tuple of file types
	filesToProcess = []

	tag = eyeD3.Tag()	

	for files in types:
		filesToProcess.extend(glob.glob(files))		

	for f in filesToProcess:
		tag.link(f)
		if useMp3TagsAsName:
			wavFileName = ntpath.split(f)[0] + os.sep + tag.getArtist() + " --- " + tag.getTitle() + ".wav"
		else:
			wavFileName = f.replace(".mp3",".wav")		
		command = "avconv -i \"" + f + "\" -ar " +str(Fs) + " -ac " + str(nC) + " \"" + wavFileName + "\"";
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

