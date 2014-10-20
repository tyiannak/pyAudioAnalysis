import sys, os
stations = {}

stations["skai"] = "http://159.253.149.12:8282"

def recordStation(stationName, outputName):
#	command = "gst-launch-1.0 uridecodebin uri="+stations["skai"]+" ! tee name=t \ 
#	t. ! queue ! audioresample ! audio/x-raw, rate=16000 ! wavenc ! filesink location=" + outputName
	command = "gst-launch-1.0 uridecodebin uri=" + stations[stationName]+ " 	! tee name=t \ t. ! queue ! audioresample ! audio/x-raw, rate=16000 ! wavenc ! filesink location="+outputName
	print command
	os.system(command)


def main(argv):
	recordStation(argv[1], argv[2])
	return 0
	
if __name__ == '__main__':
	main(sys.argv)
