from __future__ import print_function
import os
import time
import glob
import math
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from pyAudioAnalysis import utilities
from pyAudioAnalysis import audioBasicIO
from scipy.fftpack.realtransforms import dct

eps = 0.00000001

""" Time-domain audio features """


def short_term_zero_crossing_rate(frame):
    """Computes zero crossing rate of frame"""
    count = len(frame)
    count_zero = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
    return np.float64(count_zero) / np.float64(count - 1.0)


def short_term_energy(frame):
    """Computes signal energy of frame"""
    return np.sum(frame ** 2) / np.float64(len(frame))


def short_term_energy_entropy(frame, n_short_blocks=10):
    """Computes entropy of energy"""
    frame_energy = np.sum(frame ** 2)    # total frame energy
    frame_length = len(frame)
    sub_win_len = int(np.floor(frame_length / n_short_blocks))
    if frame_length != sub_win_len * n_short_blocks:
        frame = frame[0:sub_win_len * n_short_blocks]

    # sub_wins is of size [n_short_blocks x L]
    sub_wins = frame.reshape(sub_win_len, n_short_blocks, order='F').copy()

    # Compute normalized sub-frame energies:
    s = np.sum(sub_wins ** 2, axis=0) / (frame_energy + eps)

    # Compute entropy of the normalized sub-frame energies:
    entropy = -np.sum(s * np.log2(s + eps))
    return entropy


""" Frequency-domain audio features """


def stSpectralCentroidAndSpread(X, fs):
    """Computes spectral centroid of frame (given abs(FFT))"""
    ind = (np.arange(1, len(X) + 1)) * (fs/(2.0 * len(X)))

    Xt = X.copy()
    Xt = Xt / Xt.max()
    NUM = np.sum(ind * Xt)
    DEN = np.sum(Xt) + eps

    # Centroid:
    C = (NUM / DEN)

    # Spread:
    S = np.sqrt(np.sum(((ind - C) ** 2) * Xt) / DEN)

    # Normalize:
    C = C / (fs / 2.0)
    S = S / (fs / 2.0)

    return (C, S)


def stSpectralEntropy(X, n_short_blocks=10):
    """Computes the spectral entropy"""
    L = len(X)   # number of frame samples
    Eol = np.sum(X ** 2)   # total spectral energy

    sub_win_len = int(np.floor(L / n_short_blocks))  # length of sub-frame
    if L != sub_win_len * n_short_blocks:
        X = X[0:sub_win_len * n_short_blocks]

    # define sub-frames (using matrix reshape)
    sub_wins = X.reshape(sub_win_len, n_short_blocks, order='F').copy()
    # compute spectral sub-energies
    s = np.sum(sub_wins ** 2, axis=0) / (Eol + eps)
    # compute spectral entropy
    En = -np.sum(s*np.log2(s + eps))

    return En


def stSpectralFlux(X, X_prev):
    """
    Computes the spectral flux feature of the current frame
    ARGUMENTS:
        X:            the abs(fft) of the current frame
        X_prev:        the abs(fft) of the previous frame
    """
    # compute the spectral flux as the sum of square distances:
    sumX = np.sum(X + eps)
    sumPrevX = np.sum(X_prev + eps)
    F = np.sum((X / sumX - X_prev/sumPrevX) ** 2)

    return F


def stSpectralRollOff(X, c, fs):
    """Computes spectral roll-off"""
    totalEnergy = np.sum(X ** 2)
    fftLength = len(X)
    Thres = c*totalEnergy
    # Ffind the spectral rolloff as the frequency position 
    # where the respective spectral energy is equal to c*totalEnergy
    CumSum = np.cumsum(X ** 2) + eps
    [a, ] = np.nonzero(CumSum > Thres)
    if len(a) > 0:
        mC = np.float64(a[0]) / (float(fftLength))
    else:
        mC = 0.0
    return (mC)


def stHarmonic(frame, fs):
    """
    Computes harmonic ratio and pitch
    """
    M = np.round(0.016 * fs) - 1
    R = np.correlate(frame, frame, mode='full')

    g = R[len(frame)-1]
    R = R[len(frame):-1]

    # estimate m0 (as the first zero crossing of R)
    [a, ] = np.nonzero(np.diff(np.sign(R)))

    if len(a) == 0:
        m0 = len(R)-1
    else:
        m0 = a[0]
    if M > len(R):
        M = len(R) - 1

    Gamma = np.zeros((M), dtype=np.float64)
    CSum = np.cumsum(frame ** 2)
    Gamma[m0:M] = R[m0:M] / (np.sqrt((g * CSum[M:m0:-1])) + eps)

    ZCR = short_term_zero_crossing_rate(Gamma)

    if ZCR > 0.15:
        HR = 0.0
        f0 = 0.0
    else:
        if len(Gamma) == 0:
            HR = 1.0
            blag = 0.0
            Gamma = np.zeros((M), dtype=np.float64)
        else:
            HR = np.max(Gamma)
            blag = np.argmax(Gamma)

        # Get fundamental frequency:
        f0 = fs / (blag + eps)
        if f0 > 5000:
            f0 = 0.0
        if HR < 0.1:
            f0 = 0.0

    return (HR, f0)


def mfccInitFilterBanks(fs, nfft, lowfreq=133.33, linc=200/3, logsc=1.0711703,
                        numLinFiltTotal=13, numLogFilt=27):
    """
    Computes the triangular filterbank for MFCC computation 
    (used in the stFeatureExtraction function before the stMFCC function call)
    This function is taken from the scikits.talkbox library (MIT Licence):
    https://pypi.python.org/pypi/scikits.talkbox
    """

    if fs < 8000:
        nlogfil = 5

    # Total number of filters
    nFiltTotal = numLinFiltTotal + numLogFilt

    # Compute frequency points of the triangle:
    freqs = np.zeros(nFiltTotal+2)
    freqs[:numLinFiltTotal] = lowfreq + np.arange(numLinFiltTotal) * linc
    freqs[numLinFiltTotal:] = freqs[numLinFiltTotal-1] * logsc ** \
                              np.arange(1, numLogFilt + 3)
    heights = 2./(freqs[2:] - freqs[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = np.zeros((nFiltTotal, nfft))
    nfreqs = np.arange(nfft) / (1. * nfft) * fs

    for i in range(nFiltTotal):
        lowTrFreq = freqs[i]
        cenTrFreq = freqs[i+1]
        highTrFreq = freqs[i+2]

        lid = np.arange(np.floor(lowTrFreq * nfft / fs) + 1, 
                           np.floor(cenTrFreq * nfft / fs) + 1,
                           dtype=np.int)
        lslope = heights[i] / (cenTrFreq - lowTrFreq)
        rid = np.arange(np.floor(cenTrFreq * nfft / fs) + 1, 
                                       np.floor(highTrFreq * nfft / fs) + 1,
                           dtype=np.int)
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

    mspec = np.log10(np.dot(X, fbank.T)+eps)
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:n_mfcc_feats]
    return ceps


def stChromaFeaturesInit(nfft, fs):
    """
    This function initializes the chroma matrices used in the calculation
    of the chroma features
    """
    freqs = np.array([((f + 1) * fs) / (2 * nfft) for f in range(nfft)])    
    Cp = 27.50    
    nChroma = np.round(12.0 * np.log2(freqs / Cp)).astype(int)

    nFreqsPerChroma = np.zeros((nChroma.shape[0], ))

    uChroma = np.unique(nChroma)
    for u in uChroma:
        idx = np.nonzero(nChroma == u)
        nFreqsPerChroma[idx] = idx[0].shape
    
    return nChroma, nFreqsPerChroma


def stChromaFeatures(X, fs, nChroma, nFreqsPerChroma):
    #TODO: 1 complexity
    #TODO: 2 bug with large windows

    chromaNames = ['A', 'A#', 'B', 'C', 'C#', 'D', 
                   'D#', 'E', 'F', 'F#', 'G', 'G#']
    spec = X**2    
    if nChroma.max()<nChroma.shape[0]:        
        C = np.zeros((nChroma.shape[0],))
        C[nChroma] = spec
        C /= nFreqsPerChroma[nChroma]
    else:        
        I = np.nonzero(nChroma>nChroma.shape[0])[0][0]        
        C = np.zeros((nChroma.shape[0],))
        C[nChroma[0:I-1]] = spec            
        C /= nFreqsPerChroma
    finalC = np.zeros((12, 1))
    newD = int(np.ceil(C.shape[0] / 12.0) * 12)
    C2 = np.zeros((newD, ))
    C2[0:C.shape[0]] = C
    C2 = C2.reshape(int(C2.shape[0]/12), 12)
    #for i in range(12):
    #    finalC[i] = np.sum(C[i:C.shape[0]:12])
    finalC = np.matrix(np.sum(C2, axis=0)).T
    finalC /= spec.sum()

#    ax = plt.gca()
#    plt.hold(False)
#    plt.plot(finalC)
#    ax.set_xticks(range(len(chromaNames)))
#    ax.set_xticklabels(chromaNames)
#    xaxis = np.arange(0, 0.02, 0.01);
#    ax.set_yticks(range(len(xaxis)))
#    ax.set_yticklabels(xaxis)
#    plt.show(block=False)
#    plt.draw()

    return chromaNames, finalC


def stChromagram(signal, fs, win, step, PLOT=False):
    """
    Short-term FFT mag for spectogram estimation:
    Returns:
        a np array (nFFT x numOfShortTermWindows)
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
    signal = np.double(signal)
    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (np.abs(signal)).max()
    signal = (signal - DC) / (MAX - DC)

    N = len(signal)        # total number of signals
    cur_p = 0
    count_fr = 0
    nfft = int(win / 2)
    nChroma, nFreqsPerChroma = stChromaFeaturesInit(nfft, fs)
    chromaGram = np.array([], dtype=np.float64)

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
            chromaGram = np.vstack((chromaGram, C.T))
    FreqAxis = chromaNames
    TimeAxis = [(t * step) / fs for t in range(chromaGram.shape[0])]

    if (PLOT):
        fig, ax = plt.subplots()
        chromaGramToPlot = chromaGram.transpose()[::-1, :]
        Ratio = int(chromaGramToPlot.shape[1] / (3*chromaGramToPlot.shape[0]))
        if Ratio < 1:
            Ratio = 1
        chromaGramToPlot = np.repeat(chromaGramToPlot, Ratio, axis=0)
        imgplot = plt.imshow(chromaGramToPlot)
        fstep = int(nfft / 5.0)
#        FreqTicks = range(0, int(nfft) + fstep, fstep)
#        FreqTicksLabels = [str(fs/2-int((f*fs) / (2*nfft))) for f in FreqTicks]
        ax.set_yticks(range(int(Ratio / 2), len(FreqAxis) * Ratio, Ratio))
        ax.set_yticklabels(FreqAxis[::-1])
        t_step = int(count_fr / 3)
        time_ticks = range(0, count_fr, t_step)
        time_ticks_labels = ['%.2f' % (float(t * step) / fs)
                             for t in time_ticks]
        ax.set_xticks(time_ticks)
        ax.set_xticklabels(time_ticks_labels)
        ax.set_xlabel('time (secs)')
        imgplot.set_cmap('jet')
        plt.colorbar()
        plt.show()

    return (chromaGram, TimeAxis, FreqAxis)


def phormants(x, fs):
    N = len(x)
    w = np.hamming(N)

    # Apply window and high pass filter.
    x1 = x * w   
    x1 = lfilter([1], [1., 0.63], x1)
    
    # Get LPC.    
    ncoeff = 2 + fs / 1000
    A, e, k = lpc(x1, ncoeff)    
    #A, e, k = lpc(x1, 8)

    # Get roots.
    rts = np.roots(A)
    rts = [r for r in rts if np.imag(r) >= 0]

    # Get angles.
    angz = np.arctan2(np.imag(rts), np.real(rts))

    # Get frequencies.    
    frqs = sorted(angz * (fs / (2 * math.pi)))

    return frqs


def beatExtraction(st_features, win_len, PLOT=False):
    """
    This function extracts an estimate of the beat rate for a musical signal.
    ARGUMENTS:
     - st_features:     a np array (n_feats x numOfShortTermWindows)
     - win_len:        window size in seconds
    RETURNS:
     - BPM:            estimates of beats per minute
     - Ratio:          a confidence measure
    """

    # Features that are related to the beat tracking task:
    toWatch = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

    max_beat_time = int(round(2.0 / win_len))
    hist_all = np.zeros((max_beat_time,))
    # for each feature
    for ii, i in enumerate(toWatch):
        # dif threshold (3 x Mean of Difs)
        DifThres = 2.0 * (np.abs(st_features[i, 0:-1] -
                                    st_features[i, 1::])).mean()
        if DifThres<=0:
            DifThres = 0.0000000000000001
        # detect local maxima
        [pos1, _] = utilities.peakdet(st_features[i, :], DifThres)
        posDifs = []
        # compute histograms of local maxima changes
        for j in range(len(pos1)-1):
            posDifs.append(pos1[j+1]-pos1[j])
        [hist_times, HistEdges] = \
            np.histogram(posDifs, np.arange(0.5, max_beat_time + 1.5))
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
    I = np.argmax(hist_all)
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
        a np array (nFFT x numOfShortTermWindows)
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
    signal = np.double(signal)
    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (np.abs(signal)).max()
    signal = (signal - DC) / (MAX - DC)

    N = len(signal)        # total number of signals
    cur_p = 0
    count_fr = 0
    nfft = int(win / 2)
    specgram = np.array([], dtype=np.float64)

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
            specgram = np.vstack((specgram, X))

    FreqAxis = [float((f + 1) * fs) / (2 * nfft)
                for f in range(specgram.shape[1])]
    TimeAxis = [float(t * step) / fs for t in range(specgram.shape[0])]

    if (PLOT):
        fig, ax = plt.subplots()
        imgplot = plt.imshow(specgram.transpose()[::-1, :])
        fstep = int(nfft / 5.0)
        FreqTicks = range(0, int(nfft) + fstep, fstep)
        FreqTicksLabels = [str(fs / 2 - int((f * fs) /
                                            (2 * nfft))) for f in FreqTicks]
        ax.set_yticks(FreqTicks)
        ax.set_yticklabels(FreqTicksLabels)
        t_step = int(count_fr/3)
        time_ticks = range(0, count_fr, t_step)
        time_ticks_labels = \
            ['%.2f' % (float(t * step) / fs) for t in time_ticks]
        ax.set_xticks(time_ticks)
        ax.set_xticklabels(time_ticks_labels)
        ax.set_xlabel('time (secs)')
        ax.set_ylabel('freq (Hz)')
        imgplot.set_cmap('jet')
        plt.colorbar()
        plt.show()

    return (specgram, TimeAxis, FreqAxis)


""" Windowing and feature extraction """


def short_term_feature_extraction(signal, sampling_rate, window, step):
    """
    This function implements the shor-term windowing process.
    For each short-term window a set of features is extracted.
    This results to a sequence of feature vectors, stored in a np matrix.

    ARGUMENTS
        signal:       the input signal samples
        sampling_rate:           the sampling freq (in Hz)
        win:          the short-term window size (in samples)
        step:         the short-term window step (in samples)
    RETURNS
        st_features:   a np array (n_feats x numOfShortTermWindows)
    """

    window = int(window)
    step = int(step)

    # signal normalization
    signal = np.double(signal)
    signal = signal / (2.0 ** 15)
    dc_offset = signal.mean()
    signal_max = (np.abs(signal)).max()
    signal = (signal - dc_offset) / (signal_max + 0.0000000001)

    number_of_samples = len(signal)  # total number of samples
    cur_p = 0
    count_fr = 0
    nFFT = int(window / 2)

    # compute the triangular filter banks used in the mfcc calculation
    [fbank, freqs] = mfccInitFilterBanks(sampling_rate, nFFT)
    nChroma, nFreqsPerChroma = stChromaFeaturesInit(nFFT, sampling_rate)

    n_time_spectral_feats = 8
    n_harmonic_feats = 0
    n_mfcc_feats = 13
    n_chroma_feats = 13
    n_total_feats = n_time_spectral_feats + n_mfcc_feats + \
                    n_harmonic_feats + n_chroma_feats
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
    # for each short-term window to end of signal
    while (cur_p + window - 1 < number_of_samples):
        count_fr += 1
        x = signal[cur_p:cur_p + window]  # get current window
        cur_p = cur_p + step   # update window position
        X = abs(fft(x))  # get fft magnitude
        X = X[0:nFFT]  # normalize fft
        X = X / len(X)
        if count_fr == 1:
            X_prev = X.copy()  # keep previous fft mag (used in spectral flux)
        curFV = np.zeros((n_total_feats, 1))
        curFV[0] = short_term_zero_crossing_rate(x)  # zero crossing rate
        curFV[1] = short_term_energy(x)  # short-term energy
        curFV[2] = short_term_energy_entropy(x)  # short-term entropy of energy
        # sp centroid/spread
        [curFV[3], curFV[4]] = stSpectralCentroidAndSpread(X, sampling_rate)
        curFV[5] = stSpectralEntropy(X)   # spectral entropy
        curFV[6] = stSpectralFlux(X, X_prev)   # spectral flux
        curFV[7] = stSpectralRollOff(X, 0.90, sampling_rate)  # spectral rolloff
        curFV[n_time_spectral_feats:n_time_spectral_feats+n_mfcc_feats, 0] = \
            stMFCC(X, fbank, n_mfcc_feats).copy()    # MFCCs
        chromaNames, chromaF = stChromaFeatures(X, sampling_rate, nChroma,
                                                nFreqsPerChroma)
        curFV[n_time_spectral_feats + n_mfcc_feats:
              n_time_spectral_feats + n_mfcc_feats + n_chroma_feats - 1] = \
            chromaF
        curFV[n_time_spectral_feats + n_mfcc_feats + n_chroma_feats - 1] = \
            chromaF.std()
        st_features.append(curFV)
        # delta features
        """
        if count_fr>1:
            delta = curFV - prevFV
            curFVFinal = np.concatenate((curFV, delta))            
        else:
            curFVFinal = np.concatenate((curFV, curFV))
        prevFV = curFV
        st_features.append(curFVFinal)        
        """
        # end of delta
        X_prev = X.copy()

    st_features = np.concatenate(st_features, 1)
    return st_features, feature_names


def mid_term_feature_extraction(signal, sampling_rate, mid_term_window,
                                mid_term_step, short_term_window,
                                short_term_step):
    """
    Mid-term feature extraction
    """

    st_features, f_names = \
        short_term_feature_extraction(signal, sampling_rate, short_term_window,
                                      short_term_step)

    n_stats = 2
    n_feats = len(st_features)
    mt_win_ratio = int(round(mid_term_window / short_term_step))
    mt_step_ratio = int(round(mid_term_step / short_term_step))

    mt_features, mid_feature_names = [], []
    for i in range(n_stats * n_feats):
        mt_features.append([])
        mid_feature_names.append("")

    # for each of the short-term features:
    for i in range(n_feats):
        cur_position = 0
        num_st_features = len(st_features[i])
        mid_feature_names[i] = f_names[i] + "_" + "mean"
        mid_feature_names[i + n_feats] = f_names[i] + "_" + "std"

        while cur_position < num_st_features:
            end = cur_position + mt_win_ratio
            if end > num_st_features:
                end = num_st_features
            cur_st_feats = st_features[i][cur_position:end]

            mt_features[i].append(np.mean(cur_st_feats))
            mt_features[i + n_feats].append(np.std(cur_st_feats))
            cur_position += mt_step_ratio
    return np.array(mt_features), st_features, mid_feature_names


# TODO
def stFeatureSpeed(signal, fs, win, step):

    signal = np.double(signal)
    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (np.abs(signal)).max()
    signal = (signal - DC) / MAX
    # print (np.abs(signal)).max()

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
    [fbank, freqs] = mfccInitFilterBanks(fs, nfft, lowfreq, 
                                         linsc, logsc, nlinfil, nlogfil)

    n_time_spectral_feats = 8
    n_harmonic_feats = 1
    n_total_feats = n_time_spectral_feats + n_mfcc_feats + n_harmonic_feats
    #st_features = np.array([], dtype=np.float64)
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
#        M = np.round(0.016 * fs) - 1
#        R = np.correlate(frame, frame, mode='full')
        st_features.append(stHarmonic(x, fs))
#        for i in range(len(X)):
            #if (i < (len(X) / 8)) and (i > (len(X)/40)):
            #    Ex += X[i]*X[i]
            #El += X[i]*X[i]
#        st_features.append(Ex / El)
#        st_features.append(np.argmax(X))
#        if curFV[n_time_spectral_feats+n_mfcc_feats+1]>0:
#            print curFV[n_time_spectral_feats+n_mfcc_feats], curFV[n_time_
    #            spectral_feats+n_mfcc_feats+1]
    return np.array(st_features)


""" Feature Extraction Wrappers
 - The first two feature extraction wrappers are used to extract 
   long-term averaged audio features for a list of WAV files stored in a 
   given category.
   It is important to note that, one single feature is extracted per WAV 
   file (not the whole sequence of feature vectors)

 """


def dirWavFeatureExtraction(dirName, mt_win, mt_step, st_win, st_step,
                            compute_beat=False):
    """
    This function extracts the mid-term features of the WAVE files of a 
    particular folder.

    The resulting feature vector is extracted by long-term averaging the
    mid-term features.
    Therefore ONE FEATURE VECTOR is extracted for each WAV file.

    ARGUMENTS:
        - dirName:        the path of the WAVE directory
        - mt_win, mt_step:    mid-term window and step (in seconds)
        - st_win, st_step:    short-term window and step (in seconds)
    """

    all_mt_feats = np.array([])
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
        [fs, x] = audioBasicIO.read_audio_file(wavFile)
        if isinstance(x, int):
            continue        

        t1 = time.clock()        
        x = audioBasicIO.stereo_to_mono(x)
        if x.shape[0]<float(fs)/5:
            print("  (AUDIO FILE TOO SMALL - SKIPPING)")
            continue
        wav_file_list2.append(wavFile)
        if compute_beat:
            [mt_term_feats, st_features, mt_feature_names] = \
                mid_term_feature_extraction(x, fs, round(mt_win * fs),
                                            round(mt_step * fs),
                                            round(fs * st_win), 
                                            round(fs * st_step))
            [beat, beat_conf] = beatExtraction(st_features, st_step)
        else:
            [mt_term_feats, _, mt_feature_names] = \
                mid_term_feature_extraction(x, fs, round(mt_win * fs),
                                            round(mt_step * fs),
                                            round(fs * st_win), 
                                            round(fs * st_step))

        mt_term_feats = np.transpose(mt_term_feats)
        mt_term_feats = mt_term_feats.mean(axis=0)
        # long term averaging of mid-term statistics
        if (not np.isnan(mt_term_feats).any()) and \
                (not np.isinf(mt_term_feats).any()):
            if compute_beat:
                mt_term_feats = np.append(mt_term_feats, beat)
                mt_term_feats = np.append(mt_term_feats, beat_conf)
            if len(all_mt_feats) == 0:
                # append feature vector
                all_mt_feats = mt_term_feats
            else:
                all_mt_feats = np.vstack((all_mt_feats, mt_term_feats))
            t2 = time.clock()
            duration = float(len(x)) / fs
            process_times.append((t2 - t1) / duration)
    if len(process_times) > 0:
        print("Feature extraction complexity ratio: "
              "{0:.1f} x realtime".format((1.0 / 
                                           np.mean(np.array(process_times)))))
    return (all_mt_feats, wav_file_list2, mt_feature_names)


def dirsWavFeatureExtraction(dirNames, mt_win, mt_step, st_win, st_step, 
                             compute_beat=False):
    """
    Same as dirWavFeatureExtraction, but instead of a single dir it
    takes a list of paths as input and returns a list of feature matrices.
    EXAMPLE:
    [features, classNames] =
           a.dirsWavFeatureExtraction(['audioData/classSegmentsRec/noise',
                                       'audioData/classSegmentsRec/speech',
                                       'audioData/classSegmentsRec/brush-teeth',
                                       'audioData/classSegmentsRec/shower'], 1, 
                                       1, 0.02, 0.02);

    It can be used during the training process of a classification model ,
    in order to get feature matrices from various audio classes (each stored in
    a separate path)
    """

    # feature extraction for each class:
    features = []
    classNames = []
    fileNames = []
    for i, d in enumerate(dirNames):
        [f, fn, feature_names] = \
            dirWavFeatureExtraction(d, mt_win, mt_step, st_win, st_step,
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


def dirWavFeatureExtractionNoAveraging(dirName, mt_win, mt_step, st_win,
                                       st_step):
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

    all_mt_feats = np.array([])
    signal_idx = np.array([])
    process_times = []

    types = ('*.wav', '*.aif',  '*.aiff', '*.ogg')
    wav_file_list = []
    for files in types:
        wav_file_list.extend(glob.glob(os.path.join(dirName, files)))

    wav_file_list = sorted(wav_file_list)

    for i, wavFile in enumerate(wav_file_list):
        [fs, x] = audioBasicIO.read_audio_file(wavFile)
        if isinstance(x, int):
            continue        
        
        x = audioBasicIO.stereo_to_mono(x)
        [mt_term_feats, _, _] = \
            mid_term_feature_extraction(x, fs, round(mt_win * fs),
                                        round(mt_step * fs),
                                        round(fs * st_win),
                                        round(fs * st_step))

        mt_term_feats = np.transpose(mt_term_feats)
        if len(all_mt_feats) == 0:                # append feature vector
            all_mt_feats = mt_term_feats
            signal_idx = np.zeros((mt_term_feats.shape[0], ))
        else:
            all_mt_feats = np.vstack((all_mt_feats, mt_term_feats))
            signal_idx = np.append(signal_idx, i *
                                   np.ones((mt_term_feats.shape[0], )))

    return (all_mt_feats, signal_idx, wav_file_list)


# The following two feature extraction wrappers extract features for given audio
# files, however  NO LONG-TERM AVERAGING is performed. Therefore, the output for
# each audio file is NOT A SINGLE FEATURE VECTOR but a whole feature matrix.
#
# Also, another difference between the following two wrappers and the previous
# is that they NO LONG-TERM AVERAGING IS PERFORMED. In other words, the WAV
# files in these functions are not used as uniform samples that need to be
# averaged but as sequences

def mtFeatureExtractionToFile(fileName, midTermSize, midTermStep, shortTermSize,
                              shortTermStep, outPutFile, storeStFeatures=False,
                              storeToCSV=False, PLOT=False):
    """
    This function is used as a wrapper to:
    a) read the content of a WAV file
    b) perform mid-term feature extraction on that signal
    c) write the mid-term feature sequences to a np file
    """
    [fs, x] = audioBasicIO.read_audio_file(fileName)
    x = audioBasicIO.stereo_to_mono(x)
    if storeStFeatures:
        [mtF, stF, _] = mid_term_feature_extraction(x, fs,
                                                    round(fs * midTermSize),
                                                    round(fs * midTermStep),
                                                    round(fs * shortTermSize),
                                                    round(fs * shortTermStep))
    else:
        [mtF, _, _] = mid_term_feature_extraction(x, fs,
                                                  round(fs * midTermSize),
                                                  round(fs * midTermStep),
                                                  round(fs * shortTermSize),
                                                  round(fs * shortTermStep))
    # save mt features to np file
    np.save(outPutFile, mtF)
    if PLOT:
        print("Mid-term np file: " + outPutFile + ".npy saved")
    if storeToCSV:
        np.savetxt(outPutFile+".csv", mtF.T, delimiter=",")
        if PLOT:
            print("Mid-term CSV file: " + outPutFile + ".csv saved")

    if storeStFeatures:
        # save st features to np file
        np.save(outPutFile+"_st", stF)
        if PLOT:
            print("Short-term np file: " + outPutFile + "_st.npy saved")
        if storeToCSV:
            # store st features to CSV file
            np.savetxt(outPutFile+"_st.csv", stF.T, delimiter=",")
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
