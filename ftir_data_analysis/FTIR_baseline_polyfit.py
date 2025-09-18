# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 14:21:19 2025

@author: klj
"""

##uses a baseline anchor points file (txt) to automate baseline
##correction for bulk FTIR data. Works by finding minima in spectra and finding a set of 
## closest to the baseline anchor point starting values. The baseline created is 
##a polynomial fit. Remember that some distorion of peaks will be inevitable with any baseline
##correction.

#REQUIREMENTS
## - This currently requires all data to be the same spacing and range as each other, but not 
##   necessarily the same as the data it was written for. 
## - Currently configured for data that HAS replicates. Replicates in filename must be separated from
##   the rest of the file name with a single underscore (_) and must be the last character(s) before the 
##   file extension, e.g. date-fileID_1.csv, date-fileID_2.csv, etc. 
##   currently configured for csv files with x wavenumber and y intensity values, in absorbance format, 
##   with positive peaks. 

import os #needed for os.walk
from pathlib import Path 
import numpy as np
from scipy.signal import find_peaks as find_peaks
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#groups replicates together into sets - returns a list of lists. sublists contain the filenames in a set. 
#parent list contains the lists for all sets in a date folder.
#depends on files being in order by replicate (which should happen if all of file name except replicate is the same) 
def getSets(filelist):
    setList = [filelist[0]]
    sets = []
    for i in range(1, len(filelist)):
        if '_'.join(filelist[i].split('_')[:-1]) == '_'.join(setList[0].split('_')[:-1]):   #if all but replicate number matches,
            setList.append(filelist[i])                                                     #add to set list 
        else:
            sets.append(setList)                                                            #if not, add completed sub list to parent list 
            setList = [filelist[i]]                                                         #and start a new sub list 
    sets.append(setList)                                                                    #add the last sub list to the parent list
    return sets

#find the minima for each spectrum and identifies the minima closest to the given anchor points
#returns a list of indices corresponding to anchor points in the data. 
def getMins(wns, spectrum, blPts):
    invSpec = np.multiply(spectrum, -1)                                                     #take inverse of spectrum (to use peak finder function)
    specMinsAllInd, _ = find_peaks(invSpec)                                                 #from scipy.signal  - find_peaks - returns all minima indices 
    minsInd = []                                                                            #create list to hold only minima corresponding to preset anchor points 
    for point in blPts:
        tempMinInd = min(specMinsAllInd, key=lambda x:abs(point-wns[x]))                   #for each anchor point, find the index of the picked minima that is closest to that point in the data 
        minsInd.append(tempMinInd)
    return minsInd

#polynomial function, used by curve_fit
#indefinite number of args so that power can change fluidly 
#power is equal to the number of coefficients minus one  (e.g. y = Ax^2 + Bx + C, 3 coefficients, power of 2)
def polynomialFit(x, *coeffs):
     y = np.polyval(coeffs, x)
     return y
 
    
def fitBaseline(minsInd, wns, spectrum, power): 
    baselineYVal = min(spectrum[minsInd])
    spectrum = np.subtract(spectrum, baselineYVal) 
    blYVals = spectrum[minsInd].tolist()
    #interpolate end baseline point
    m = (spectrum[minsInd[0]]-spectrum[minsInd[1]])/(minsInd[0]-minsInd[1])    
    endBl = (-1)*m*(minsInd[0])+spectrum[minsInd[0]]
    
    
    #add new end bl point to sets
    minsInd.insert(0,0)
    blYVals.insert(0, endBl)
    
    numCoeff = power + 1
    p0=np.ones(numCoeff)
    popt, pcov = curve_fit(polynomialFit, minsInd, blYVals, p0=p0)
    blXVals = np.linspace(min(minsInd), max(minsInd), len(wns))
    baseline = polynomialFit(blXVals, *popt)
    
    return spectrum, baseline, minsInd, blYVals

#just plots the original spectrum, baseline, and new baseline corrected spec, for monitoring. 
def plotBaseline(wns, spectrum, baseline, blCorr, minsInd, blYVals, file):
    figSpectrum, specX = plt.subplots(figsize=(10,5), dpi=100)
    specX.axis((400, 4000, -0.05, 0.4))
    specX.xaxis.set_inverted(True)
    specLine, = specX.plot(wns, spectrum, c='red', label=str(file), linewidth=0.5)
    blLine, = specX.plot(wns, baseline, c='black', label='baseline', linewidth=0.5)
    corrLine, = specX.plot(wns, blCorr, c='blue', label=str(str(file)+" - corrected"), linewidth=0.5)
    specX.plot(wns[minsInd], blYVals, 'o', label='bl anchor points')
    specX.legend()
    return

#baseline loop
def baselinePolyFitLoop(
        rawDataDir, 
        polyFitPower = 3, 
        blOutFolderNm = "1_baseline-corrected_TEST",
        blFileNm='PET-baseline-wns-fit.txt',
        showPlots=False,
        ):
    
    #create baseline output main folder
    rawDataFolder = Path(rawDataDir)
    parentDir = rawDataFolder.absolute().parent 
    blCorrFolder = parentDir / blOutFolderNm
    blCorrFolder.mkdir(parents=True, exist_ok=True)
    
    #get baseline anchor points file. 
    blFilePath = parentDir / blFileNm      #this is just the file with the starting anchor points. 
    baselinePoints = np.loadtxt(blFilePath, delimiter=" ")
    
    totalFiles = 0
    for root, dirs, files in os.walk(rawDataFolder):
        for file in files: 
            totalFiles +=1
    fileCounter = 0
    print(f'total files for baseline correction: {totalFiles}')
    
    #looping through files to find file level
    for root, dirs, files in os.walk(rawDataFolder):
        depth = root.replace(str(rawDataFolder), '').count(os.sep)
        outputFolder = Path(root.replace(str(rawDataFolder), str(blCorrFolder)))
        outputFolder.mkdir(parents=True, exist_ok=True)
        
        for filename in files: 
            #gets wavenumber array if not already in existence. 
            if 'wavenumbers' not in locals():
                wavenumbers = np.loadtxt(Path(root) / filename, delimiter=',')[:,0]
                break
        
        if len(files) != 0: 
            #grouping files by replicate
            replicateSets = getSets(files)
            for fileSet in replicateSets:
                dataSet = np.array([])
                for filename in fileSet:
                    #print(filename)
                    fileCounter +=1
                    #print(filename)
                    print(f'\r processing {str(fileCounter)} out of {str(totalFiles)} files, {str(round((fileCounter/totalFiles *100), 1))}% ', end=' ')
                    rawSpectrum = np.loadtxt(Path(root) / filename, delimiter=',')[:,1]
                    specMinsInd = getMins(wavenumbers, rawSpectrum, baselinePoints)
                    #shiftSpec, blSpec, blXInds, blYIntVals = makeBaseline(specMinsInd, wavenumbers, rawSpectrum)
                    shiftSpec, blSpec, blXInds, blYIntVals = fitBaseline(specMinsInd, wavenumbers, rawSpectrum, polyFitPower)
                    blCorrSpec = shiftSpec - blSpec
                    
                    #plotting uses a lot of memory so I would only use this while your set is small just to check on it.
                    if showPlots == True:
                        plotBaseline(wavenumbers, shiftSpec, blSpec, blCorrSpec, blXInds, blYIntVals, filename)
                    
                    #starts a new BL corrected set if it's the first one in a set. adds to the existing one otherwise.
                    if np.any(dataSet)==False:
                          dataSet = blCorrSpec
                    else:
                          dataSet = np.column_stack((dataSet, blCorrSpec))
            
            outputFile = np.column_stack((wavenumbers, dataSet))
            #creates output path for each new file. 
            outputFilepath = outputFolder / str('_'.join(filename.split('_')[:-1]) + '_blCorr.csv')



if __name__ == "__main__":          #does not run if importing only if running
    baselinePolyFitLoop("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-exposure/0_raw-data")    
    
    
        
#print(depth)
