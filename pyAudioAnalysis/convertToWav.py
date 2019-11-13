# USAGE:
# convertToWav <folder path> <sampling rate> <number of channels>
#

import glob, sys, os

def getVideoFilesFromFolder(dirPath):
	types = (dirPath+os.sep+'*.avi', dirPath+os.sep+'*.mkv', dirPath+os.sep+'*.mp4', dirPath+os.sep+'*.mp3', dirPath+os.sep+'*.flac', dirPath+os.sep+'*.ogg') # the tuple of file types
	files_grabbed = []
	for files in types:
		files_grabbed.extend(glob.glob(files))
	return files_grabbed

def main(argv):
	if (len(argv)==4):
		files = getVideoFilesFromFolder(argv[1])
		samplingRate = int(argv[2])
		channels = int(argv[3])
	
		for f in files:
			ffmpegString = 'avconv -i ' + '\"' + f + '\"' + ' -ar ' + str(samplingRate) + ' -ac ' + str(channels) + ' ' + '\"' + os.path.splitext(f)[0] + '\"' + '.wav' 
			os.system(ffmpegString)

if __name__ == '__main__':
	main(sys.argv)
