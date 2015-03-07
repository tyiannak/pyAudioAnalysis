#from pyAudioAnalysis import audioSegmentation as aS
#[flagsInd, classesAll, acc] = aS.mtFileClassification("data/scottish.wav", "data/svmSM", "svm", True, 'data/scottish.segments')


from pyAudioAnalysis import audioSegmentation as aS
aS.trainHMM_fromFile('radioFinal/train/bbc4A.wav', 'radioFinal/train/bbc4A.segments', 'hmmTemp1', 1.0, 1.0)	# train using a single file
aS.trainHMM_fromDir('radioFinal/small/', 'hmmTemp2', 1.0, 1.0)							# train using a set of files in a folder
aS.hmmSegmentation('data/scottish.wav', 'hmmTemp1', True, 'data/scottish.segments')				# test 1
aS.hmmSegmentation('data/scottish.wav', 'hmmTemp2', True, 'data/scottish.segments')				# test 2

