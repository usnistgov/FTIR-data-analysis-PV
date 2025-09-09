# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os 
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks as find_peaks
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)
folderNm='FTIR-data-PET-exposure'  
parentDir = Path().absolute().parent
blFTIRFolder = parentDir / folderNm / "1_baseline-corrected" 
filterListPath = parentDir / "filters-pct-T.csv"
normWn = '723'

specCounter = 0


def getSpec(fileNm):
    file = np.loadtxt(dateFolder / fileNm, delimiter=',')
    wns = file[:,0]
    allSpec = np.delete(file, 0, 1)
    return wns, allSpec


def getNormPeak(wns, spectrum):
    specPksAllInd, _ = find_peaks(spectrum)
    normPkInd = min(specPksAllInd, key=lambda x:abs(wns[x]-float(normWn)))
    return normPkInd

def getNormSpectrum(normPkInd, wns, spectrum): 
    normDiv = spectrum[normPkInd]
    normSpec = np.divide(spectrum, normDiv)
    
    return normSpec

def plotNormSpec(wns, spectrum, normPkInd, file):
    figSpectrum, specX = plt.subplots(figsize=(10,5))
    specLine, = specX.plot(wns, spectrum, c='red', label=str(file), linewidth=0.5)
    specX.plot(wns[normPkInd], spectrum[normPkInd], 'x', label=str(wns[normPeakInd]))
    specX.legend()
    return

def plotCheck(wns, spectrum):
    figSpectrum, specX = plt.subplots(figsize=(10,5))
    specLine, = specX.plot(wns, spectrum, c='red', label='check', linewidth=0.5)

#normalize loop
for chFolder in blFTIRFolder.iterdir():
    chamberOutFolder = parentDir / folderNm / "2_normalized" / str(normWn) /str(str(chFolder).split('\\')[-1:][0])
    for dateFolder in chFolder.iterdir():
        dateOutFolder = chamberOutFolder / str(str(str(dateFolder).split('\\')[-1:][0]))
        dateOutFolder.mkdir(parents=True, exist_ok=True)
        for filename in os.listdir(dateFolder):
            print(filename)
            wavenumbers, blCorrSet = getSpec(filename)
            rows, columns = blCorrSet.shape
            normSet = np.array([])
            for i in range(columns):
                specCounter+=1
                blCorrSpec = blCorrSet[:,i]
                normPeakInd = getNormPeak(wavenumbers, blCorrSpec)
                normSpectrum = getNormSpectrum(normPeakInd, wavenumbers, blCorrSpec)
                #plotNormSpec(wavenumbers, normSpectrum, normPeakInd, filename)
                if np.any(normSet)==False:
                    normSet = normSpectrum
                else:
                    normSet = np.column_stack((normSet, normSpectrum))
            normAvg = normSet.mean(axis=1)

            outputSetFile = np.column_stack((wavenumbers, normSet))
            outputAvgFile = np.column_stack((wavenumbers, normAvg))
            
            outputSetPath = dateOutFolder / str(str(filename[:-10].replace(" ", ""))+"N" +str(normWn)+".csv")
            outputAvgPath = dateOutFolder / str(str(filename[:-10].replace(" ", ""))+"N" +str(normWn)+"_Avg.csv")    

            np.savetxt(outputSetPath, outputSetFile, '%5.7f', delimiter=',')
            np.savetxt(outputAvgPath, outputAvgFile, '%5.7f', delimiter=',')

print(specCounter)       
