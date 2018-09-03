from __future__ import print_function
import time
import os
import glob
import numpy
import math
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct
import matplotlib.pyplot as plt
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import utilities
from scipy.signal import lfilter

eps = 0.00000001

""" Time-domain audio features """


def stZCR(frame):
    """Computes zero crossing rate of frame"""
    count = len(frame)
    countZ = numpy.sum(numpy.abs(numpy.diff(numpy.sign(frame)))) / 2
    return (numpy.float64(countZ) / numpy.float64(count-1.0))


def stEnergy(frame):
    """Computes signal energy of frame"""
    return numpy.sum(frame ** 2) / numpy.float64(len(frame))


def stEnergyEntropy(frame, n_short_blocks=10):
    """Computes entropy of energy"""
    Eol = numpy.sum(frame ** 2)    # total frame energy
    L = len(frame)
    sub_win_len = int(numpy.floor(L / n_short_blocks))
    if L != sub_win_len * n_short_blocks:
            frame = frame[0:sub_win_len * n_short_blocks]
    # sub_wins is of size [n_short_blocks x L]
    sub_wins = frame.reshape(sub_win_len, n_short_blocks, order='F').copy()

    # Compute normalized sub-frame energies:
    s = numpy.sum(sub_wins ** 2, axis=0) / (Eol + eps)

    # Compute entropy of the normalized sub-frame energies:
    Entropy = -numpy.sum(s * numpy.log2(s + eps))
    return Entropy


""" Frequency-domain audio features """


def stSpectralCentroidAndSpread(X, fs):
    """Computes spectral centroid of frame (given abs(FFT))"""
    ind = (numpy.arange(1, len(X) + 1)) * (fs/(2.0 * len(X)))

    Xt = X.copy()
    Xt = Xt / Xt.max()
    NUM = numpy.sum(ind * Xt)
    DEN = numpy.sum(Xt) + eps

    # Centroid:
    C = (NUM / DEN)

    # Spread:
    S = numpy.sqrt(numpy.sum(((ind - C) ** 2) * Xt) / DEN)

    # Normalize:
    C = C / (fs / 2.0)
    S = S / (fs / 2.0)

    return (C, S)


def stSpectralEntropy(X, n_short_blocks=10):
    """Computes the spectral entropy"""
    L = len(X)                         # number of frame samples
    Eol = numpy.sum(X ** 2)            # total spectral energy

    sub_win_len = int(numpy.floor(L / n_short_blocks))   # length of sub-frame
    if L != sub_win_len * n_short_blocks:
        X = X[0:sub_win_len * n_short_blocks]

    sub_wins = X.reshape(sub_win_len, n_short_blocks, order='F').copy()  # define sub-frames (using matrix reshape)
    s = numpy.sum(sub_wins ** 2, axis=0) / (Eol + eps)                      # compute spectral sub-energies
    En = -numpy.sum(s*numpy.log2(s + eps))                                    # compute spectral entropy

    return En


def stSpectralFlux(X, X_prev):
    """
    Computes the spectral flux feature of the current frame
    ARGUMENTS:
        X:            the abs(fft) of the current frame
        X_prev:        the abs(fft) of the previous frame
    """
    # compute the spectral flux as the sum of square distances:
    sumX = numpy.sum(X + eps)
    sumPrevX = numpy.sum(X_prev + eps)
    F = numpy.sum((X / sumX - X_prev/sumPrevX) ** 2)

    return F


def stSpectralRollOff(X, c, fs):
    """Computes spectral roll-off"""
    totalEnergy = numpy.sum(X ** 2)
    fftLength = len(X)
    Thres = c*totalEnergy
    # Ffind the spectral rolloff as the frequency position 
    # where the respective spectral energy is equal to c*totalEnergy
    CumSum = numpy.cumsum(X ** 2) + eps
    [a, ] = numpy.nonzero(CumSum > Thres)
    if len(a) > 0:
        mC = numpy.float64(a[0]) / (float(fftLength))
    else:
        mC = 0.0
    return (mC)


def stHarmonic(frame, fs):
    """
    Computes harmonic ratio and pitch
    """
    M = numpy.round(0.016 * fs) - 1
    R = numpy.correlate(frame, frame, mode='full')

    g = R[len(frame)-1]
    R = R[len(frame):-1]

    # estimate m0 (as the first zero crossing of R)
    [a, ] = numpy.nonzero(numpy.diff(numpy.sign(R)))

    if len(a) == 0:
        m0 = len(R)-1
    else:
        m0 = a[0]
    if M > len(R):
        M = len(R) - 1

    Gamma = numpy.zeros((M), dtype=numpy.float64)
    CSum = numpy.cumsum(frame ** 2)
    Gamma[m0:M] = R[m0:M] / (numpy.sqrt((g * CSum[M:m0:-1])) + eps)

    ZCR = stZCR(Gamma)

    if ZCR > 0.15:
        HR = 0.0
        f0 = 0.0
    else:
        if len(Gamma) == 0:
            HR = 1.0
            blag = 0.0
            Gamma = numpy.zeros((M), dtype=numpy.float64)
        else:
            HR = numpy.max(Gamma)
            blag = numpy.argmax(Gamma)

        # Get fundamental frequency:
        f0 = fs / (blag + eps)
        if f0 > 5000:
            f0 = 0.0
        if HR < 0.1:
            f0 = 0.0

    return (HR, f0)


def mfccInitFilterBanks(fs, nfft):
    """
    Computes the triangular filterbank for MFCC computation 
    (used in the stFeatureExtraction function before the stMFCC function call)
    This function is taken from the scikits.talkbox library (MIT Licence):
    https://pypi.python.org/pypi/scikits.talkbox
    """

    # filter bank params:
    lowfreq = 133.33
    linsc = 200/3.
    logsc = 1.0711703
    numLinFiltTotal = 13
    numLogFilt = 27

    if fs < 8000:
        nlogfil = 5

    # Total number of filters
    nFiltTotal = numLinFiltTotal + numLogFilt

    # Compute frequency points of the triangle:
    freqs = numpy.zeros(nFiltTotal+2)
    freqs[:numLinFiltTotal] = lowfreq + numpy.arange(numLinFiltTotal) * linsc
    freqs[numLinFiltTotal:] = freqs[numLinFiltTotal-1] * logsc ** numpy.arange(1, numLogFilt + 3)
    heights = 2./(freqs[2:] - freqs[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = numpy.zeros((nFiltTotal, nfft))
    nfreqs = numpy.arange(nfft) / (1. * nfft) * fs

    for i in range(nFiltTotal):
        lowTrFreq = freqs[i]
        cenTrFreq = freqs[i+1]
        highTrFreq = freqs[i+2]

        lid = numpy.arange(numpy.floor(lowTrFreq * nfft / fs) + 1, 
                           numpy.floor(cenTrFreq * nfft / fs) + 1,  
                                       dtype=numpy.int)
        lslope = heights[i] / (cenTrFreq - lowTrFreq)
        rid = numpy.arange(numpy.floor(cenTrFreq * nfft / fs) + 1, 
                                       numpy.floor(highTrFreq * nfft / fs) + 1, 
                                       dtype=numpy.int)
        rslope = heights[i] / (highTrFreq - cenTrFreq)
        fbank[i][lid] = lslope * (nfreqs[lid] - lowTrFreq)
        fbank[i][rid] = rslope * (highTrFreq - nfreqs[rid])

    return fbank, freqs


def stMFCC(X, fbank, n_mfcc_feats):
    """
    Computes the MFCCs of a frame, given the fft mag

    ARGUMENTS:
        X:        fft magnitude abs(FFT)
        fbank:    filter bank (see mfccInitFilterBanks)
    RETURN
        ceps:     MFCCs (13 element vector)

    Note:    MFCC calculation is, in general, taken from the 
             scikits.talkbox library (MIT Licence),
    #    with a small number of modifications to make it more 
         compact and suitable for the pyAudioAnalysis Lib
    """

    mspec = numpy.log10(numpy.dot(X, fbank.T)+eps)
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:n_mfcc_feats]
    return ceps


def stChromaFeaturesInit(nfft, fs):
    """
    This function initializes the chroma matrices used in the calculation of the chroma features
    """
    freqs = numpy.array([((f + 1) * fs) / (2 * nfft) for f in range(nfft)])    
    Cp = 27.50    
    nChroma = numpy.round(12.0 * numpy.log2(freqs / Cp)).astype(int)

    nFreqsPerChroma = numpy.zeros((nChroma.shape[0], ))

    uChroma = numpy.unique(nChroma)
    for u in uChroma:
        idx = numpy.nonzero(nChroma == u)
        nFreqsPerChroma[idx] = idx[0].shape
    
    return nChroma, nFreqsPerChroma


def stChromaFeatures(X, fs, nChroma, nFreqsPerChroma):
    #TODO: 1 complexity
    #TODO: 2 bug with large windows

    chromaNames = ['A', 'A#', 'B', 'C', 'C#', 'D', 
                   'D#', 'E', 'F', 'F#', 'G', 'G#']
    spec = X**2    
    if nChroma.max()<nChroma.shape[0]:        
        C = numpy.zeros((nChroma.shape[0],))
        C[nChroma] = spec
        C /= nFreqsPerChroma[nChroma]
    else:        
        I = numpy.nonzero(nChroma>nChroma.shape[0])[0][0]        
        C = numpy.zeros((nChroma.shape[0],))
        C[nChroma[0:I-1]] = spec            
        C /= nFreqsPerChroma
    finalC = numpy.zeros((12, 1))
    newD = int(numpy.ceil(C.shape[0] / 12.0) * 12)
    C2 = numpy.zeros((newD, ))
    C2[0:C.shape[0]] = C
    C2 = C2.reshape(int(C2.shape[0]/12), 12)
    #for i in range(12):
    #    finalC[i] = numpy.sum(C[i:C.shape[0]:12])
    finalC = numpy.matrix(numpy.sum(C2, axis=0)).T
    finalC /= spec.sum()

#    ax = plt.gca()
#    plt.hold(False)
#    plt.plot(finalC)
#    ax.set_xticks(range(len(chromaNames)))
#    ax.set_xticklabels(chromaNames)
#    xaxis = numpy.arange(0, 0.02, 0.01);
#    ax.set_yticks(range(len(xaxis)))
#    ax.set_yticklabels(xaxis)
#    plt.show(block=False)
#    plt.draw()

    return chromaNames, finalC


def stChromagram(signal, fs, win, step, PLOT=False):
    """
    Short-term FFT mag for spectogram estimation:
    Returns:
        a numpy array (nFFT x numOfShortTermWindows)
    ARGUMENTS:
        signal:      the input signal samples
        fs:          the sampling freq (in Hz)
        win:         the short-term window size (in samples)
        step:        the short-term window step (in samples)
        PLOT:        flag, 1 if results are to be ploted
    RETURNS:
    """
    win = int(win)
    step = int(step)
    signal = numpy.double(signal)
    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC) / (MAX - DC)

    N = len(signal)        # total number of signals
    cur_p = 0
    count_fr = 0
    nfft = int(win / 2)
    nChroma, nFreqsPerChroma = stChromaFeaturesInit(nfft, fs)
    chromaGram = numpy.array([], dtype=numpy.float64)

    while (cur_p + win - 1 < N):
        count_fr += 1
        x = signal[cur_p:cur_p + win]
        cur_p = cur_p + step
        X = abs(fft(x))
        X = X[0:nfft]
        X = X / len(X)
        chromaNames, C = stChromaFeatures(X, fs, nChroma, nFreqsPerChroma)
        C = C[:, 0]
        if count_fr == 1:
            chromaGram = C.T
        else:
            chromaGram = numpy.vstack((chromaGram, C.T))
    FreqAxis = chromaNames
    TimeAxis = [(t * step) / fs for t in range(chromaGram.shape[0])]

    if (PLOT):
        fig, ax = plt.subplots()
        chromaGramToPlot = chromaGram.transpose()[::-1, :]
        Ratio = int(chromaGramToPlot.shape[1] / (3*chromaGramToPlot.shape[0]))
        if Ratio < 1:
            Ratio = 1
        chromaGramToPlot = numpy.repeat(chromaGramToPlot, Ratio, axis=0)
        imgplot = plt.imshow(chromaGramToPlot)
        fstep = int(nfft / 5.0)
#        FreqTicks = range(0, int(nfft) + fstep, fstep)
#        FreqTicksLabels = [str(fs/2-int((f*fs) / (2*nfft))) for f in FreqTicks]
        ax.set_yticks(range(int(Ratio / 2), len(FreqAxis) * Ratio, Ratio))
        ax.set_yticklabels(FreqAxis[::-1])
        TStep = int(count_fr / 3)
        TimeTicks = range(0, count_fr, TStep)
        TimeTicksLabels = ['%.2f' % (float(t * step) / fs) for t in TimeTicks]
        ax.set_xticks(TimeTicks)
        ax.set_xticklabels(TimeTicksLabels)
        ax.set_xlabel('time (secs)')
        imgplot.set_cmap('jet')
        plt.colorbar()
        plt.show()

    return (chromaGram, TimeAxis, FreqAxis)


def phormants(x, fs):
    N = len(x)
    w = numpy.hamming(N)

    # Apply window and high pass filter.
    x1 = x * w   
    x1 = lfilter([1], [1., 0.63], x1)
    
    # Get LPC.    
    ncoeff = 2 + fs / 1000
    A, e, k = lpc(x1, ncoeff)    
    #A, e, k = lpc(x1, 8)

    # Get roots.
    rts = numpy.roots(A)
    rts = [r for r in rts if numpy.imag(r) >= 0]

    # Get angles.
    angz = numpy.arctan2(numpy.imag(rts), numpy.real(rts))

    # Get frequencies.    
    frqs = sorted(angz * (fs / (2 * math.pi)))

    return frqs
def beatExtraction(st_features, win_len, PLOT=False):
    """
    This function extracts an estimate of the beat rate for a musical signal.
    ARGUMENTS:
     - st_features:     a numpy array (n_feats x numOfShortTermWindows)
     - win_len:        window size in seconds
    RETURNS:
     - BPM:            estimates of beats per minute
     - Ratio:          a confidence measure
    """

    # Features that are related to the beat tracking task:
    toWatch = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

    max_beat_time = int(round(2.0 / win_len))
    hist_all = numpy.zeros((max_beat_time,))
    for ii, i in enumerate(toWatch):                                        # for each feature
        DifThres = 2.0 * (numpy.abs(st_features[i, 0:-1] - st_features[i, 1::])).mean()    # dif threshold (3 x Mean of Difs)
        if DifThres<=0:
            DifThres = 0.0000000000000001        
        [pos1, _] = utilities.peakdet(st_features[i, :], DifThres)           # detect local maxima
        posDifs = []                                                        # compute histograms of local maxima changes
        for j in range(len(pos1)-1):
            posDifs.append(pos1[j+1]-pos1[j])
        [hist_times, HistEdges] = numpy.histogram(posDifs, numpy.arange(0.5, max_beat_time + 1.5))
        hist_centers = (HistEdges[0:-1] + HistEdges[1::]) / 2.0
        hist_times = hist_times.astype(float) / st_features.shape[1]
        hist_all += hist_times
        if PLOT:
            plt.subplot(9, 2, ii + 1)
            plt.plot(st_features[i, :], 'k')
            for k in pos1:
                plt.plot(k, st_features[i, k], 'k*')
            f1 = plt.gca()
            f1.axes.get_xaxis().set_ticks([])
            f1.axes.get_yaxis().set_ticks([])

    if PLOT:
        plt.show(block=False)
        plt.figure()

    # Get beat as the argmax of the agregated histogram:
    I = numpy.argmax(hist_all)
    bpms = 60 / (hist_centers * win_len)
    BPM = bpms[I]
    # ... and the beat ratio:
    Ratio = hist_all[I] / hist_all.sum()

    if PLOT:
        # filter out >500 beats from plotting:
        hist_all = hist_all[bpms < 500]
        bpms = bpms[bpms < 500]

        plt.plot(bpms, hist_all, 'k')
        plt.xlabel('Beats per minute')
        plt.ylabel('Freq Count')
        plt.show(block=True)

    return BPM, Ratio


def stSpectogram(signal, fs, win, step, PLOT=False):
    """
    Short-term FFT mag for spectogram estimation:
    Returns:
        a numpy array (nFFT x numOfShortTermWindows)
    ARGUMENTS:
        signal:      the input signal samples
        fs:          the sampling freq (in Hz)
        win:         the short-term window size (in samples)
        step:        the short-term window step (in samples)
        PLOT:        flag, 1 if results are to be ploted
    RETURNS:
    """
    win = int(win)
    step = int(step)
    signal = numpy.double(signal)
    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC) / (MAX - DC)

    N = len(signal)        # total number of signals
    cur_p = 0
    count_fr = 0
    nfft = int(win / 2)
    specgram = numpy.array([], dtype=numpy.float64)

    while (cur_p + win - 1 < N):
        count_fr += 1
        x = signal[cur_p:cur_p+win]
        cur_p = cur_p + step
        X = abs(fft(x))
        X = X[0:nfft]
        X = X / len(X)

        if count_fr == 1:
            specgram = X ** 2
        else:
            specgram = numpy.vstack((specgram, X))

    FreqAxis = [float((f + 1) * fs) / (2 * nfft) for f in range(specgram.shape[1])]
    TimeAxis = [float(t * step) / fs for t in range(specgram.shape[0])]

    if (PLOT):
        fig, ax = plt.subplots()
        imgplot = plt.imshow(specgram.transpose()[::-1, :])
        fstep = int(nfft / 5.0)
        FreqTicks = range(0, int(nfft) + fstep, fstep)
        FreqTicksLabels = [str(fs / 2 - int((f * fs) / (2 * nfft))) for f in FreqTicks]
        ax.set_yticks(FreqTicks)
        ax.set_yticklabels(FreqTicksLabels)
        TStep = int(count_fr/3)
        TimeTicks = range(0, count_fr, TStep)
        TimeTicksLabels = ['%.2f' % (float(t * step) / fs) for t in TimeTicks]
        ax.set_xticks(TimeTicks)
        ax.set_xticklabels(TimeTicksLabels)
        ax.set_xlabel('time (secs)')
        ax.set_ylabel('freq (Hz)')
        imgplot.set_cmap('jet')
        plt.colorbar()
        plt.show()

    return (specgram, TimeAxis, FreqAxis)


""" Windowing and feature extraction """


def stFeatureExtraction(signal, fs, win, step):
    """
    This function implements the shor-term windowing process. For each short-term window a set of features is extracted.
    This results to a sequence of feature vectors, stored in a numpy matrix.

    ARGUMENTS
        signal:       the input signal samples
        fs:           the sampling freq (in Hz)
        win:          the short-term window size (in samples)
        step:         the short-term window step (in samples)
    RETURNS
        st_features:   a numpy array (n_feats x numOfShortTermWindows)
    """

    win = int(win)
    step = int(step)

    # Signal normalization
    signal = numpy.double(signal)

    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC) / (MAX + 0.0000000001)

    N = len(signal)                                # total number of samples
    cur_p = 0
    count_fr = 0
    nFFT = int(win / 2)

    [fbank, freqs] = mfccInitFilterBanks(fs, nFFT)                # compute the triangular filter banks used in the mfcc calculation
    nChroma, nFreqsPerChroma = stChromaFeaturesInit(nFFT, fs)

    n_time_spectral_feats = 8
    n_harmonic_feats = 0
    n_mfcc_feats = 13
    n_chroma_feats = 13
    n_total_feats = n_time_spectral_feats + n_mfcc_feats + n_harmonic_feats + n_chroma_feats
#    n_total_feats = n_time_spectral_feats + n_mfcc_feats + n_harmonic_feats
    feature_names = []
    feature_names.append("zcr")
    feature_names.append("energy")
    feature_names.append("energy_entropy")
    feature_names += ["spectral_centroid", "spectral_spread"]
    feature_names.append("spectral_entropy")
    feature_names.append("spectral_flux")
    feature_names.append("spectral_rolloff")
    feature_names += ["mfcc_{0:d}".format(mfcc_i) 
                      for mfcc_i in range(1, n_mfcc_feats+1)]
    feature_names += ["chroma_{0:d}".format(chroma_i) 
                      for chroma_i in range(1, n_chroma_feats)]
    feature_names.append("chroma_std")
    st_features = []
    while (cur_p + win - 1 < N):                        # for each short-term window until the end of signal
        count_fr += 1
        x = signal[cur_p:cur_p+win]                    # get current window
        cur_p = cur_p + step                           # update window position
        X = abs(fft(x))                                  # get fft magnitude
        X = X[0:nFFT]                                    # normalize fft
        X = X / len(X)
        if count_fr == 1:
            X_prev = X.copy()                             # keep previous fft mag (used in spectral flux)
        curFV = numpy.zeros((n_total_feats, 1))
        curFV[0] = stZCR(x)                              # zero crossing rate
        curFV[1] = stEnergy(x)                           # short-term energy
        curFV[2] = stEnergyEntropy(x)                    # short-term entropy of energy
        [curFV[3], curFV[4]] = stSpectralCentroidAndSpread(X, fs)    # spectral centroid and spread
        curFV[5] = stSpectralEntropy(X)                  # spectral entropy
        curFV[6] = stSpectralFlux(X, X_prev)              # spectral flux
        curFV[7] = stSpectralRollOff(X, 0.90, fs)        # spectral rolloff
        curFV[n_time_spectral_feats:n_time_spectral_feats+n_mfcc_feats, 0] = \
            stMFCC(X, fbank, n_mfcc_feats).copy()    # MFCCs
        chromaNames, chromaF = stChromaFeatures(X, fs, nChroma, nFreqsPerChroma)
        curFV[n_time_spectral_feats + n_mfcc_feats:
              n_time_spectral_feats + n_mfcc_feats + n_chroma_feats - 1] = \
            chromaF
        curFV[n_time_spectral_feats + n_mfcc_feats + n_chroma_feats - 1] = \
            chromaF.std()
        st_features.append(curFV)
        # delta features
        '''
        if count_fr>1:
            delta = curFV - prevFV
            curFVFinal = numpy.concatenate((curFV, delta))            
        else:
            curFVFinal = numpy.concatenate((curFV, curFV))
        prevFV = curFV
        st_features.append(curFVFinal)        
        '''
        # end of delta
        X_prev = X.copy()

    st_features = numpy.concatenate(st_features, 1)
    return st_features, feature_names


def mtFeatureExtraction(signal, fs, mt_win, mt_step, st_win, st_step):
    """
    Mid-term feature extraction
    """

    mt_win_ratio = int(round(mt_win / st_step))
    mt_step_ratio = int(round(mt_step / st_step))

    mt_features = []

    st_features, f_names = stFeatureExtraction(signal, fs, st_win, st_step)
    n_feats = len(st_features)
    n_stats = 2

    mt_features, mid_feature_names = [], []
    #for i in range(n_stats * n_feats + 1):
    for i in range(n_stats * n_feats):
        mt_features.append([])
        mid_feature_names.append("")

    for i in range(n_feats):        # for each of the short-term features:
        cur_p = 0
        N = len(st_features[i])
        mid_feature_names[i] = f_names[i] + "_" + "mean"
        mid_feature_names[i + n_feats] = f_names[i] + "_" + "std"

        while (cur_p < N):
            N1 = cur_p
            N2 = cur_p + mt_win_ratio
            if N2 > N:
                N2 = N
            cur_st_feats = st_features[i][N1:N2]

            mt_features[i].append(numpy.mean(cur_st_feats))
            mt_features[i + n_feats].append(numpy.std(cur_st_feats))
            #mt_features[i+2*n_feats].append(numpy.std(cur_st_feats) / (numpy.mean(cur_st_feats)+0.00000010))
            cur_p += mt_step_ratio
    return numpy.array(mt_features), st_features, mid_feature_names


# TODO
def stFeatureSpeed(signal, fs, win, step):

    signal = numpy.double(signal)
    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (numpy.abs(signal)).max()
    signal = (signal - DC) / MAX
    # print (numpy.abs(signal)).max()

    N = len(signal)        # total number of signals
    cur_p = 0
    count_fr = 0

    lowfreq = 133.33
    linsc = 200/3.
    logsc = 1.0711703
    nlinfil = 13
    nlogfil = 27
    n_mfcc_feats = 13
    nfil = nlinfil + nlogfil
    nfft = win / 2
    if fs < 8000:
        nlogfil = 5
        nfil = nlinfil + nlogfil
        nfft = win / 2

    # compute filter banks for mfcc:
    [fbank, freqs] = mfccInitFilterBanks(fs, nfft, lowfreq, linsc, logsc, nlinfil, nlogfil)

    n_time_spectral_feats = 8
    n_harmonic_feats = 1
    n_total_feats = n_time_spectral_feats + n_mfcc_feats + n_harmonic_feats
    #st_features = numpy.array([], dtype=numpy.float64)
    st_features = []

    while (cur_p + win - 1 < N):
        count_fr += 1
        x = signal[cur_p:cur_p + win]
        cur_p = cur_p + step
        X = abs(fft(x))
        X = X[0:nfft]
        X = X / len(X)
        Ex = 0.0
        El = 0.0
        X[0:4] = 0
#        M = numpy.round(0.016 * fs) - 1
#        R = numpy.correlate(frame, frame, mode='full')
        st_features.append(stHarmonic(x, fs))
#        for i in range(len(X)):
            #if (i < (len(X) / 8)) and (i > (len(X)/40)):
            #    Ex += X[i]*X[i]
            #El += X[i]*X[i]
#        st_features.append(Ex / El)
#        st_features.append(numpy.argmax(X))
#        if curFV[n_time_spectral_feats+n_mfcc_feats+1]>0:
#            print curFV[n_time_spectral_feats+n_mfcc_feats], curFV[n_time_spectral_feats+n_mfcc_feats+1]
    return numpy.array(st_features)


""" Feature Extraction Wrappers

 - The first two feature extraction wrappers are used to extract long-term averaged
   audio features for a list of WAV files stored in a given category.
   It is important to note that, one single feature is extracted per WAV file (not the whole sequence of feature vectors)

 """


def dirWavFeatureExtraction(dirName, mt_win, mt_step, st_win, st_step,
                            compute_beat=False):
    """
    This function extracts the mid-term features of the WAVE files of a particular folder.

    The resulting feature vector is extracted by long-term averaging the mid-term features.
    Therefore ONE FEATURE VECTOR is extracted for each WAV file.

    ARGUMENTS:
        - dirName:        the path of the WAVE directory
        - mt_win, mt_step:    mid-term window and step (in seconds)
        - st_win, st_step:    short-term window and step (in seconds)
    """

    all_mt_feats = numpy.array([])
    process_times = []

    types = ('*.wav', '*.aif',  '*.aiff', '*.mp3', '*.au', '*.ogg')
    wav_file_list = []
    for files in types:
        wav_file_list.extend(glob.glob(os.path.join(dirName, files)))

    wav_file_list = sorted(wav_file_list)    
    wav_file_list2, mt_feature_names = [], []
    for i, wavFile in enumerate(wav_file_list):        
        print("Analyzing file {0:d} of "
              "{1:d}: {2:s}".format(i+1,
                                    len(wav_file_list),
                                    wavFile))
        if os.stat(wavFile).st_size == 0:
            print("   (EMPTY FILE -- SKIPPING)")
            continue        
        [fs, x] = audioBasicIO.readAudioFile(wavFile)
        if isinstance(x, int):
            continue        

        t1 = time.clock()        
        x = audioBasicIO.stereo2mono(x)
        if x.shape[0]<float(fs)/5:
            print("  (AUDIO FILE TOO SMALL - SKIPPING)")
            continue
        wav_file_list2.append(wavFile)
        if compute_beat:
            [mt_term_feats, st_features, mt_feature_names] = \
                mtFeatureExtraction(x, fs, round(mt_win * fs),
                                    round(mt_step * fs),
                                    round(fs * st_win), round(fs * st_step))
            [beat, beat_conf] = beatExtraction(st_features, st_step)
        else:
            [mt_term_feats, _, mt_feature_names] = \
                mtFeatureExtraction(x, fs, round(mt_win * fs),
                                    round(mt_step * fs),
                                    round(fs * st_win), round(fs * st_step))

        mt_term_feats = numpy.transpose(mt_term_feats)
        mt_term_feats = mt_term_feats.mean(axis=0)
        # long term averaging of mid-term statistics
        if (not numpy.isnan(mt_term_feats).any()) and \
                (not numpy.isinf(mt_term_feats).any()):
            if compute_beat:
                mt_term_feats = numpy.append(mt_term_feats, beat)
                mt_term_feats = numpy.append(mt_term_feats, beat_conf)
            if len(all_mt_feats) == 0:
                # append feature vector
                all_mt_feats = mt_term_feats
            else:
                all_mt_feats = numpy.vstack((all_mt_feats, mt_term_feats))
            t2 = time.clock()
            duration = float(len(x)) / fs
            process_times.append((t2 - t1) / duration)
    if len(process_times) > 0:
        print("Feature extraction complexity ratio: "
              "{0:.1f} x realtime".format((1.0 / numpy.mean(numpy.array(process_times)))))
    return (all_mt_feats, wav_file_list2, mt_feature_names)


def dirsWavFeatureExtraction(dirNames, mt_win, mt_step, st_win, st_step, compute_beat=False):
    '''
    Same as dirWavFeatureExtraction, but instead of a single dir it
    takes a list of paths as input and returns a list of feature matrices.
    EXAMPLE:
    [features, classNames] =
           a.dirsWavFeatureExtraction(['audioData/classSegmentsRec/noise','audioData/classSegmentsRec/speech',
                                       'audioData/classSegmentsRec/brush-teeth','audioData/classSegmentsRec/shower'], 1, 1, 0.02, 0.02);

    It can be used during the training process of a classification model ,
    in order to get feature matrices from various audio classes (each stored in a seperate path)
    '''

    # feature extraction for each class:
    features = []
    classNames = []
    fileNames = []
    for i, d in enumerate(dirNames):
        [f, fn, feature_names] = dirWavFeatureExtraction(d, mt_win, mt_step,
                                                         st_win, st_step,
                                                         compute_beat=compute_beat)
        if f.shape[0] > 0:
            # if at least one audio file has been found in the provided folder:
            features.append(f)
            fileNames.append(fn)
            if d[-1] == os.sep:
                classNames.append(d.split(os.sep)[-2])
            else:
                classNames.append(d.split(os.sep)[-1])
    return features, classNames, fileNames


def dirWavFeatureExtractionNoAveraging(dirName, mt_win, mt_step, st_win, st_step):
    """
    This function extracts the mid-term features of the WAVE
    files of a particular folder without averaging each file.

    ARGUMENTS:
        - dirName:          the path of the WAVE directory
        - mt_win, mt_step:    mid-term window and step (in seconds)
        - st_win, st_step:    short-term window and step (in seconds)
    RETURNS:
        - X:                A feature matrix
        - Y:                A matrix of file labels
        - filenames:
    """

    all_mt_feats = numpy.array([])
    signal_idx = numpy.array([])
    process_times = []

    types = ('*.wav', '*.aif',  '*.aiff', '*.ogg')
    wav_file_list = []
    for files in types:
        wav_file_list.extend(glob.glob(os.path.join(dirName, files)))

    wav_file_list = sorted(wav_file_list)

    for i, wavFile in enumerate(wav_file_list):
        [fs, x] = audioBasicIO.readAudioFile(wavFile)
        if isinstance(x, int):
            continue        
        
        x = audioBasicIO.stereo2mono(x)
        [mt_term_feats, _, _] = mtFeatureExtraction(x, fs, round(mt_win * fs),
                                                    round(mt_step * fs),
                                                    round(fs * st_win),
                                                    round(fs * st_step))

        mt_term_feats = numpy.transpose(mt_term_feats)
        if len(all_mt_feats) == 0:                # append feature vector
            all_mt_feats = mt_term_feats
            signal_idx = numpy.zeros((mt_term_feats.shape[0], ))
        else:
            all_mt_feats = numpy.vstack((all_mt_feats, mt_term_feats))
            signal_idx = numpy.append(signal_idx, i * numpy.ones((mt_term_feats.shape[0], )))

    return (all_mt_feats, signal_idx, wav_file_list)


# The following two feature extraction wrappers extract features for given audio files, however
# NO LONG-TERM AVERAGING is performed. Therefore, the output for each audio file is NOT A SINGLE FEATURE VECTOR
# but a whole feature matrix.
#
# Also, another difference between the following two wrappers and the previous is that they NO LONG-TERM AVERAGING IS PERFORMED.
# In other words, the WAV files in these functions are not used as uniform samples that need to be averaged but as sequences

def mtFeatureExtractionToFile(fileName, midTermSize, midTermStep, shortTermSize, shortTermStep, outPutFile,
                              storeStFeatures=False, storeToCSV=False, PLOT=False):
    """
    This function is used as a wrapper to:
    a) read the content of a WAV file
    b) perform mid-term feature extraction on that signal
    c) write the mid-term feature sequences to a numpy file
    """
    [fs, x] = audioBasicIO.readAudioFile(fileName)
    x = audioBasicIO.stereo2mono(x)
    if storeStFeatures:
        [mtF, stF, _] = mtFeatureExtraction(x, fs,
                                         round(fs * midTermSize),
                                         round(fs * midTermStep),
                                         round(fs * shortTermSize),
                                         round(fs * shortTermStep))
    else:
        [mtF, _, _] = mtFeatureExtraction(x, fs, round(fs*midTermSize),
                                       round(fs * midTermStep),
                                       round(fs * shortTermSize),
                                       round(fs * shortTermStep))
    # save mt features to numpy file
    numpy.save(outPutFile, mtF)
    if PLOT:
        print("Mid-term numpy file: " + outPutFile + ".npy saved")
    if storeToCSV:
        numpy.savetxt(outPutFile+".csv", mtF.T, delimiter=",")
        if PLOT:
            print("Mid-term CSV file: " + outPutFile + ".csv saved")

    if storeStFeatures:
        # save st features to numpy file
        numpy.save(outPutFile+"_st", stF)
        if PLOT:
            print("Short-term numpy file: " + outPutFile + "_st.npy saved")
        if storeToCSV:
            # store st features to CSV file
            numpy.savetxt(outPutFile+"_st.csv", stF.T, delimiter=",")
            if PLOT:
                print("Short-term CSV file: " + outPutFile + "_st.csv saved")


def mtFeatureExtractionToFileDir(dirName, midTermSize, midTermStep,
                                 shortTermSize, shortTermStep,
                                 storeStFeatures=False, storeToCSV=False,
                                 PLOT=False):
    types = (dirName + os.sep + '*.wav', )
    filesToProcess = []
    for files in types:
        filesToProcess.extend(glob.glob(files))
    for f in filesToProcess:
        outPath = f
        mtFeatureExtractionToFile(f, midTermSize, midTermStep, shortTermSize,
                                  shortTermStep, outPath, storeStFeatures,
                                  storeToCSV, PLOT)
