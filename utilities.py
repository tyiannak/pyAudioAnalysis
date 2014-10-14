import sys, os, numpy

def isfloat(x):
	"""
	Check if argument is float
	"""
	try:
		a = float(x)
	except ValueError:
		return False
	else:
		return True

def isint(x):
	"""
	Check if argument is int
	"""
	try:
		a = float(x)
		b = int(a)
	except ValueError:
		return False
	else:
		return a == b

def isNum(x):
	"""
	Check if string argument is numerical
	"""
	return isfloat(x) or isint(x)


def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05
    % This function is released to the public domain; Any use is allowed.
    
    """
    maxtab = []
    mintab = []
       
    if x is None:
        x = numpy.arange(len(v))
    
    v = numpy.asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not numpy.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = numpy.Inf, -numpy.Inf
    mnpos, mxpos = numpy.NaN, numpy.NaN
    
    lookformax = True
    
    for i in numpy.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append(mxpos)
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append(mnpos)
                mx = this
                mxpos = x[i]
                lookformax = True
 
    return numpy.array(maxtab), numpy.array(mintab)

