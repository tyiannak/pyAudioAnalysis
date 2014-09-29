import os, sys, mlpy, shutil, struct, simplejson
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from pylab import *
import ntpath
import audioFeatureExtraction as aF	
import audioTrainTest as aT

def generateColorMap():
	'''
	This function generates a 256 jet colormap of HTML-like hex string colors (e.g. FF88AA)
	'''
	Map = cm.jet(np.arange(256))
	stringColors = []
	for i in range(Map.shape[0]):
		rgb = (int(255*Map[i][0]), int(255*Map[i][1]), int(255*Map[i][2]))
		stringColors.append(struct.pack('BBB',*rgb).encode('hex'))
	return stringColors

def levenshtein(str1, s2):
	'''
	Distance between two strings
	'''
	N1 = len(str1)
	N2 = len(s2)

	stringRange = [range(N1 + 1)] * (N2 + 1)
	for i in range(N2 + 1):
		stringRange[i] = range(i,i + N1 + 1)
	for i in range(0,N2):
		for j in range(0,N1):
			if str1[j] == s2[i]:
				stringRange[i+1][j+1] = min(stringRange[i+1][j] + 1, stringRange[i][j+1] + 1, stringRange[i][j])
			else:
				stringRange[i+1][j+1] = min(stringRange[i+1][j] + 1, stringRange[i][j+1] + 1, stringRange[i][j] + 1)
	return stringRange[N2][N1]

def textListToColors(names):
	'''
	Generates a list of colors based on a list of names (strings). Similar strings correspond to similar colors.
	'''
	# STEP A: compute strings distance between all combnations of strings
	Dnames = np.zeros( (len(names), len(names)) )
	for i in range(len(names)):
		for j in range(len(names)):
			Dnames[i,j] = 1 - 2.0 * levenshtein(names[i], names[j]) / float(len(names[i]+names[j]))

	# STEP B: pca dimanesionality reduction to a single-dimension (from the distance space)
	pca = mlpy.PCA(method='cov') 
	pca.learn(Dnames)
	coeff = pca.coeff()
	
	# STEP C: mapping of 1-dimensional values to colors in a jet-colormap
	textToColor = pca.transform(Dnames, k=1)
	textToColor = 255 * (textToColor - textToColor.min()) / (textToColor.max() - textToColor.min())
	textmaps = generateColorMap();
	colors = [textmaps[int(c)] for c in textToColor]
	return colors

def chordialDiagram(fileStr, SM, names):
	'''
	Generates a d3js chordial diagram that illustrates similarites
	'''

	colors = textListToColors(names)
	SM2 = SM.copy()
	for i in range(SM2.shape[0]):
#		M = 0.85
		a = np.sort(SM2[i,:])[::-1]
		print a
		M = np.mean(a[0:int(SM2.shape[1]/10+1)])
		print M
		SM2[i,SM2[i,:]<M] = 0;
	SM2 = SM2 + SM2.T
	print SM2
	dirChordial = fileStr + "_Chordial"
	if not os.path.isdir(dirChordial):
		os.mkdir(dirChordial)
	jsonPath 		= dirChordial + os.sep + "matrix.json"
	namesPath		= dirChordial + os.sep + "Names.csv"
 
	jsonSMMatrix = simplejson.dumps(SM2.tolist())
	f = open(jsonPath,'w'); f.write(jsonSMMatrix);  f.close()
	f = open(namesPath,'w'); f.write("name,color\n"); 
	for i, n in enumerate(names):
		f.write("{0:s},{1:s}\n".format(n,"#"+colors[i]))
	f.close()

	shutil.copyfile("similarities.html", dirChordial+os.sep+"similarities.html")
	shutil.copyfile("style.css", dirChordial+os.sep+"style.css")

def visualizeFeaturesFolder(folder):
	allMtFeatures, wavFilesList = aF.dirWavFeatureExtraction(folder, 10.0, 10.0, 0.050, 0.050)
	(F, MEAN, STD) = aT.normalizeFeatures(np.matrix(allMtFeatures))
	F = np.concatenate(F)
	pca = mlpy.PCA(method='cov') # pca (eigenvalue decomposition)
	pca.learn(F)
	coeff = pca.coeff()
	finalDims = pca.transform(F, k=2)

	for i in range(finalDims.shape[0]):			
		plt.text(finalDims[i,0], finalDims[i,1], ntpath.basename(wavFilesList[i].replace('.wav','')), horizontalalignment='center', verticalalignment='center', fontsize=10)
		plt.plot(finalDims[i,0], finalDims[i,1], '*r')
	plt.xlim([1.2*finalDims[:,0].min(), 1.2*finalDims[:,0].max()])
	plt.ylim([1.2*finalDims[:,1].min(), 1.2*finalDims[:,1].max()])			
	plt.show()
#	return finalDims, wavFilesList
	SM = 1.0 - distance.squareform(distance.pdist(finalDims, 'cosine'))
	for i in range(SM.shape[0]):
		SM[i,i] = 0.0;
	namesToVisualize = [ntpath.basename(w).replace('.wav','') for w in wavFilesList];
	chordialDiagram("visualization", SM, namesToVisualize)

