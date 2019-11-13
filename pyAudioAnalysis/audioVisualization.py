from __future__ import print_function
import shutil, struct, simplejson
from scipy.spatial import distance
from pylab import *
import ntpath
from pyAudioAnalysis import audioFeatureExtraction as aF
from pyAudioAnalysis import audioTrainTest as aT
import sklearn
import sklearn.discriminant_analysis
import os
import sys


def generateColorMap():
    '''
    This function generates a 256 jet colormap of HTML-like
    hex string colors (e.g. FF88AA)
    '''
    Map = cm.jet(np.arange(256))
    stringColors = []
    for i in range(Map.shape[0]):
        rgb = (int(255*Map[i][0]), int(255*Map[i][1]), int(255*Map[i][2]))
        if (sys.version_info > (3, 0)):
            stringColors.append((struct.pack('BBB', *rgb).hex())) # python 3
        else:
            stringColors.append(
                struct.pack('BBB', *rgb).encode('hex'))  # python2

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
                stringRange[i+1][j+1] = min(stringRange[i+1][j] + 1,
                                            stringRange[i][j+1] + 1,
                                            stringRange[i][j])
            else:
                stringRange[i+1][j+1] = min(stringRange[i+1][j] + 1,
                                            stringRange[i][j+1] + 1,
                                            stringRange[i][j] + 1)
    return stringRange[N2][N1]


def text_list_to_colors(names):
    '''
    Generates a list of colors based on a list of names (strings). Similar strings correspond to similar colors.
    '''
    # STEP A: compute strings distance between all combnations of strings
    Dnames = np.zeros( (len(names), len(names)) )
    for i in range(len(names)):
        for j in range(len(names)):
            Dnames[i,j] = 1 - 2.0 * levenshtein(names[i], names[j]) / float(len(names[i]+names[j]))

    # STEP B: pca dimanesionality reduction to a single-dimension (from the distance space)
    pca = sklearn.decomposition.PCA(n_components = 1)
    pca.fit(Dnames)    
    
    # STEP C: mapping of 1-dimensional values to colors in a jet-colormap
    textToColor = pca.transform(Dnames)
    textToColor = 255 * (textToColor - textToColor.min()) / (textToColor.max() - textToColor.min())
    textmaps = generateColorMap();
    colors = [textmaps[int(c)] for c in textToColor]
    return colors


def text_list_to_colors_simple(names):
    '''
    Generates a list of colors based on a list of names (strings). Similar strings correspond to similar colors. 
    '''
    uNames = list(set(names))
    uNames.sort()
    textToColor = [ uNames.index(n) for n in names ]
    textToColor = np.array(textToColor)
    textToColor = 255 * (textToColor - textToColor.min()) / \
                  (textToColor.max() - textToColor.min())
    textmaps = generateColorMap();
    colors = [textmaps[int(c)] for c in textToColor]
    return colors


def chordialDiagram(fileStr, SM, Threshold, names, namesCategories):
    '''
    Generates a d3js chordial diagram that illustrates similarites
    '''
    colors = text_list_to_colors_simple(namesCategories)
    SM2 = SM.copy()
    SM2 = (SM2 + SM2.T) / 2.0
    for i in range(SM2.shape[0]):
        M = Threshold
#        a = np.sort(SM2[i,:])[::-1]
#        M = np.mean(a[0:int(SM2.shape[1]/3+1)])
        SM2[i, SM2[i, :] < M] = 0;
    dirChordial = fileStr + "_Chordial"
    if not os.path.isdir(dirChordial):
        os.mkdir(dirChordial)
    jsonPath         = dirChordial + os.sep + "matrix.json"
    namesPath        = dirChordial + os.sep + "Names.csv"
 
    jsonSMMatrix = simplejson.dumps(SM2.tolist())
    f = open(jsonPath,'w'); f.write(jsonSMMatrix);  f.close()
    f = open(namesPath,'w'); f.write("name,color\n"); 
    for i, n in enumerate(names):
        f.write("{0:s},{1:s}\n".format(n,"#"+str(colors[i])))
    f.close()

    shutil.copyfile(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "data", "similarities.html"),
                    dirChordial+os.sep+"similarities.html")
    shutil.copyfile(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "data",
                                 "style.css"),
                    dirChordial+os.sep+"style.css")


def visualizeFeaturesFolder(folder, dimReductionMethod, priorKnowledge = "none"):
    '''
    This function generates a chordial visualization for the recordings of the provided path.
    ARGUMENTS:
        - folder:        path of the folder that contains the WAV files to be processed
        - dimReductionMethod:    method used to reduce the dimension of the initial feature space before computing the similarity.
        - priorKnowledge:    if this is set equal to "artist"
    '''
    if dimReductionMethod=="pca":
        allMtFeatures, wavFilesList, _ = aF.dirWavFeatureExtraction(folder, 30.0, 30.0, 0.050, 0.050, compute_beat = True)
        if allMtFeatures.shape[0]==0:
            print("Error: No data found! Check input folder")
            return
        
        namesCategoryToVisualize = [ntpath.basename(w).replace('.wav','').split(" --- ")[0] for w in wavFilesList]; 
        namesToVisualize       = [ntpath.basename(w).replace('.wav','') for w in wavFilesList]; 

        (F, MEAN, STD) = aT.normalizeFeatures([allMtFeatures])
        F = np.concatenate(F)
        
        # check that the new PCA dimension is at most equal to the number of samples
        K1 = 2
        K2 = 10
        if K1 > F.shape[0]:
            K1 = F.shape[0]
        if K2 > F.shape[0]:
            K2 = F.shape[0]
        pca1 = sklearn.decomposition.PCA(n_components = K1)
        pca1.fit(F)        
        pca2 = sklearn.decomposition.PCA(n_components = K2)
        pca2.fit(F)        

        finalDims = pca1.transform(F)
        finalDims2 = pca2.transform(F)
    else:    
        allMtFeatures, Ys, wavFilesList = aF.dirWavFeatureExtractionNoAveraging(folder, 20.0, 5.0, 0.040, 0.040) # long-term statistics cannot be applied in this context (LDA needs mid-term features)
        if allMtFeatures.shape[0]==0:
            print("Error: No data found! Check input folder")
            return
        
        namesCategoryToVisualize = [ntpath.basename(w).replace('.wav','').split(" --- ")[0] for w in wavFilesList]; 
        namesToVisualize       = [ntpath.basename(w).replace('.wav','') for w in wavFilesList]; 

        ldaLabels = Ys
        if priorKnowledge=="artist":
            uNamesCategoryToVisualize = list(set(namesCategoryToVisualize))
            YsNew = np.zeros( Ys.shape )
            for i, uname in enumerate(uNamesCategoryToVisualize):        # for each unique artist name:
                indicesUCategories = [j for j, x in enumerate(namesCategoryToVisualize) if x == uname]
                for j in indicesUCategories:
                    indices = np.nonzero(Ys==j)
                    YsNew[indices] = i
            ldaLabels = YsNew

        (F, MEAN, STD) = aT.normalizeFeatures([allMtFeatures])
        F = np.array(F[0])

        clf = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=10)
        clf.fit(F, ldaLabels)    
        reducedDims =  clf.transform(F)

        pca = sklearn.decomposition.PCA(n_components = 2)
        pca.fit(reducedDims)
        reducedDims = pca.transform(reducedDims)

        # TODO: CHECK THIS ... SHOULD LDA USED IN SEMI-SUPERVISED ONLY????

        uLabels = np.sort(np.unique((Ys)))        # uLabels must have as many labels as the number of wavFilesList elements
        reducedDimsAvg = np.zeros( (uLabels.shape[0], reducedDims.shape[1] ) )
        finalDims = np.zeros( (uLabels.shape[0], 2) ) 
        for i, u in enumerate(uLabels):
            indices = [j for j, x in enumerate(Ys) if x == u]
            f = reducedDims[indices, :]
            finalDims[i, :] = f.mean(axis=0)
        finalDims2 = reducedDims

    for i in range(finalDims.shape[0]):            
        plt.text(finalDims[i,0], finalDims[i,1], ntpath.basename(wavFilesList[i].replace('.wav','')), horizontalalignment='center', verticalalignment='center', fontsize=10)
        plt.plot(finalDims[i,0], finalDims[i,1], '*r')
    plt.xlim([1.2*finalDims[:,0].min(), 1.2*finalDims[:,0].max()])
    plt.ylim([1.2*finalDims[:,1].min(), 1.2*finalDims[:,1].max()])            
    plt.show()

    SM = 1.0 - distance.squareform(distance.pdist(finalDims2, 'cosine'))
    for i in range(SM.shape[0]):
        SM[i,i] = 0.0;


    chordialDiagram("visualization", SM, 0.50, namesToVisualize, namesCategoryToVisualize)

    SM = 1.0 - distance.squareform(distance.pdist(F, 'cosine'))
    for i in range(SM.shape[0]):
        SM[i,i] = 0.0;
    chordialDiagram("visualizationInitial", SM, 0.50, namesToVisualize, namesCategoryToVisualize)

    # plot super-categories (i.e. artistname
    uNamesCategoryToVisualize = sort(list(set(namesCategoryToVisualize)))
    finalDimsGroup = np.zeros( (len(uNamesCategoryToVisualize), finalDims2.shape[1] ) )
    for i, uname in enumerate(uNamesCategoryToVisualize):
        indices = [j for j, x in enumerate(namesCategoryToVisualize) if x == uname]
        f = finalDims2[indices, :]
        finalDimsGroup[i, :] = f.mean(axis=0)

    SMgroup = 1.0 - distance.squareform(distance.pdist(finalDimsGroup, 'cosine'))
    for i in range(SMgroup.shape[0]):
        SMgroup[i,i] = 0.0;
    chordialDiagram("visualizationGroup", SMgroup, 0.50, uNamesCategoryToVisualize, uNamesCategoryToVisualize)



