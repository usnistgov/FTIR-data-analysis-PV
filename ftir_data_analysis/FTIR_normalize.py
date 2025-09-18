# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#outputs to another folder in the same directory as the raw data folder, called 2_normalized


import os 
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks as find_peaks
import matplotlib.pyplot as plt



#retrieves the csv file containing the replicates for each sample. splits into wn col and data columns
def getSpec(fileNm, dateFold):
    file = np.loadtxt(dateFold / fileNm, delimiter=',')
    wns = file[:,0]
    allSpec = np.delete(file, 0, 1)
    return wns, allSpec

#picks peaks and find the index of the peak closest to the specified wavenumber peak (normWn)
def getNormPeak(wns, spectrum, nWn):
    specPksAllInd, _ = find_peaks(spectrum)
    normPkInd = min(specPksAllInd, key=lambda x:abs(wns[x]-float(nWn)))
    return normPkInd

#divides by the value of the normalization peak
def getNormSpectrum(normPkInd, wns, spectrum): 
    normDiv = spectrum[normPkInd]
    normSpec = np.divide(spectrum, normDiv)
    
    return normSpec

#plots normalization to do a quick visual check 
def plotNormSpec(wns, spectrum, normPkInd, file):
    figSpectrum, specX = plt.subplots(figsize=(10,5))
    specLine, = specX.plot(wns, spectrum, c='red', label=str(file), linewidth=0.5)
    specX.plot(wns[normPkInd], spectrum[normPkInd], 'x', label=str(wns[normPkInd]))
    specX.legend()
    return

# def plotCheck(wns, spectrum):
#     figSpectrum, specX = plt.subplots(figsize=(10,5))
#     specLine, = specX.plot(wns, spectrum, c='red', label='check', linewidth=0.5)

#normalize loop

def normalizeLoop(
        directory,
        folderNm='FTIR-data-PET-exposure',
        blFTIRFileNm = "1_baseline-corrected",
        normWn = '723',
        filtersFileNm = "filters-pct-T.csv"
        ):
    specCounter = 0
    fileCounter = 0
    np.set_printoptions(suppress=True)
    parentDir=Path(directory)
    blFTIRFolder = parentDir / folderNm / blFTIRFileNm
    filterListPath = parentDir / filtersFileNm
    
    for chFolder in blFTIRFolder.iterdir():
        for dateFolder in chFolder.iterdir():
            for filename in os.listdir(dateFolder):
                fileCounter +=1
    #just for progress tracking:
    totalFiles = fileCounter 
    fileCounter = 0
    print(totalFiles)
    
    for chFolder in blFTIRFolder.iterdir():
        #creates output folder 
        chamberOutFolder = parentDir / folderNm / "2_normalized" / str(normWn) /str(str(chFolder).split('\\')[-1:][0])
        for dateFolder in chFolder.iterdir():
            #creates date output folder 
            dateOutFolder = chamberOutFolder / str(str(str(dateFolder).split('\\')[-1:][0]))
            dateOutFolder.mkdir(parents=True, exist_ok=True)
            for filename in os.listdir(dateFolder):
                #print(filename)
                fileCounter+=1
                print(f'\r processing {str(fileCounter)} out of {str(totalFiles)} files, {str(round((fileCounter/totalFiles *100), 1))}% ', end=' ')
                wavenumbers, blCorrSet = getSpec(filename, dateFolder)
                rows, columns = blCorrSet.shape
                normSet = np.array([])
                for i in range(columns):
                    specCounter+=1
                    blCorrSpec = blCorrSet[:,i]
                    normPeakInd = getNormPeak(wavenumbers, blCorrSpec, normWn)
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
    
    print(f'\n Done. \n total individual spectra: {str(specCounter)}')       

if __name__ == "__main__":          #does not run if importing only if running
    normalizeLoop("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity")