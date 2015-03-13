#from pyAudioAnalysis import audioSegmentation as aS
#[flagsInd, classesAll, acc] = aS.mtFileClassification("data/scottish.wav", "data/svmSM", "svm", True, 'data/scottish.segments')


#from pyAudioAnalysis import audioSegmentation as aS
#aS.trainHMM_fromFile('radioFinal/train/bbc4A.wav', 'radioFinal/train/bbc4A.segments', 'hmmTemp1', 1.0, 1.0)	# train using a single file
#aS.trainHMM_fromDir('radioFinal/small/', 'hmmTemp2', 1.0, 1.0)							# train using a set of files in a folder
#aS.hmmSegmentation('data/scottish.wav', 'hmmTemp1', True, 'data/scottish.segments')				# test 1
#aS.hmmSegmentation('data/scottish.wav', 'hmmTemp2', True, 'data/scottish.segments')				# test 2


import audioFeatureExtraction as aF	
import audioTrainTest as aT
import audioSegmentation as aS
import audioVisualization as aV
import audioBasicIO
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import (generate_binary_structure,
									  iterate_structure, binary_erosion)
from scipy.ndimage.filters import maximum_filter
from operator import itemgetter
DEFAULT_FAN_VALUE = 15

PEAK_NEIGHBORHOOD_SIZE = 10


def get_2D_peaks(arr2D, plot=False, amp_min=10):
	# http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.morphology.iterate_structure.html#scipy.ndimage.morphology.iterate_structure
	struct = generate_binary_structure(2, 1)
	neighborhood = iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)

	# find local maxima using our fliter shape
	local_max = maximum_filter(arr2D, footprint=neighborhood) == arr2D
	background = (arr2D == 0)
	eroded_background = binary_erosion(background, structure=neighborhood,
									   border_value=1)

	# Boolean mask of arr2D with True at peaks
	detected_peaks = local_max - eroded_background

	# extract peaks
	amps = arr2D[detected_peaks]
	j, i = np.where(detected_peaks)

	# filter peaks
	amps = amps.flatten()
	peaks = zip(i, j, amps)

	peaks_filtered = [x for x in peaks if x[2] > amp_min]  # freq, time, amp

	# get indices for frequency and time
	frequency_idx = [x[1] for x in peaks_filtered]
	time_idx = [x[0] for x in peaks_filtered]

	if plot:
		# scatter of the peaks
		fig, ax = plt.subplots()
		ax.imshow(arr2D)
		ax.scatter(time_idx, frequency_idx)
		ax.set_xlabel('Time')
		ax.set_ylabel('Frequency')
		ax.set_title("Spectrogram")
#		for i in zip(frequency_idx, time_idx):
#			print i
		plt.gca().invert_yaxis()
		plt.show()

	return zip(frequency_idx, time_idx)



wavFileName = 'Blur --- Song 2_thumb1.wav'
[Fs, x] = audioBasicIO.readAudioFile(wavFileName)
x = audioBasicIO.stereo2mono(x)
specgram, TimeAxis, FreqAxis = aF.stSpectogram(x, Fs, round(Fs*0.050), round(Fs*0.050), False)
specgram = - 10 * np.log10(specgram.T)
print specgram.shape

IDX_FREQ_I = 0
IDX_TIME_J = 1
MIN_HASH_TIME_DELTA = 0
MAX_HASH_TIME_DELTA = 200

peaks = get_2D_peaks(specgram, plot=True, amp_min=20)
peaks.sort(key=itemgetter(1))
for i in range(len(peaks)):
	for j in range(1, DEFAULT_FAN_VALUE):
		if (i + j) < len(peaks):                
			freq1 = peaks[i][IDX_FREQ_I]
			freq2 = peaks[i + j][IDX_FREQ_I]
			t1 = peaks[i][IDX_TIME_J]
			t2 = peaks[i + j][IDX_TIME_J]
			t_delta = t2 - t1
			if t_delta >= MIN_HASH_TIME_DELTA and t_delta <= MAX_HASH_TIME_DELTA:
				print str(freq1), str(freq2), str(t_delta)
			#h = hashlib.sha1(
                        #"%s|%s|%s" % (str(freq1), str(freq2), str(t_delta)))
#print len(local_maxima)


