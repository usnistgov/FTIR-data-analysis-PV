# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 13:52:09 2025

@author: klj
"""

import os 
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks as find_peaks
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


specCounter = 0

#dateFolderNm = "20250110-0h" 
#dateFolderNm = "20250116-12h"
#dateFolderNm = "20250117-25h"
#dateFolderNm = "20250122-45h"
#dateFolderNm = "20250124-63h"
#dateFolderNm = "20250130-81h"
#dateFolderNm = "20250205-104h"
#dateFolderNm = "20250213-145h"
#dateFolderNm = "20250226-185h"
#dateFolderNm = "20250312-226h"
#dateFolderNm = "20250321-277h"
#dateFolderNm = "20250404-327h"
#dateFolderNm = "20250421-400h"
#dateFolderNm = "20250512-564h"
#dateFolderNm = "20250521-605h"
#dateFolderNm = "20250528-694h"
#dateFolderNm = "20250616-759h"
#dateFolderNm = "20250707-871h"
#dateFolderNm = "20250724-1095h"
#dateFolderNm = "20250804-1171h"
#dateFolderNm = "20250821-1290h"
dateFolderNm = "20250902-1393h"

np.set_printoptions(suppress=True)
folderNm='FTIR-data-PET-exposure'  #name of folder where FTIR data is stored - this will contain all downstream outputs as well in individual subfolders
parentDir = Path().absolute().parent
blFilePath = parentDir / folderNm / 'PET-baseline-wns-fit.txt'      #this is just the file with the starting anchor points. 
rawFTIRFolder = parentDir / folderNm / "0_raw-data"             #all raw data (CSV only) placed here 
wnMidFolder = "chamber-1"                                       #this needs to be one of the chambers being used, but won't need to be changed, it's only for obtaining the wn column.

#obtains the wavenumber column and baseline file data using the folders and files named above. 
def getWnsandBL():
    wnFile = os.listdir(rawFTIRFolder / wnMidFolder / str((os.listdir(rawFTIRFolder / wnMidFolder)[0])))[0]
    wnFileLoc = (rawFTIRFolder / wnMidFolder / str((os.listdir(rawFTIRFolder / wnMidFolder)[0]))) / wnFile
    wns = np.loadtxt(wnFileLoc, delimiter=',')[:,0]
    blPoints = np.loadtxt(blFilePath, delimiter=" ")
    print(blPoints)
    return wns , blPoints

#makes a list of all replciate file names for each sample measurement. 
def getSets(folder):
    setList = [os.listdir(folder)[0]]
    sets = []
    for i in range(1, len(os.listdir(folder))):    
        if os.listdir(folder)[i][:-6] == setList[0][:-6]:
            setList.append(os.listdir(folder)[i])
        else: 
            sets.append(setList)
            setList = [os.listdir(folder)[i]]
    sets.append(setList)
    return sets 

#obtains the data given a filename and date folder. 
def getSpec(dateFolder, fileNm):
    rawSpec = np.loadtxt(dateFolder / fileNm, delimiter=',')[:,1]
    return rawSpec

#find the minima for each spectrum and identifies the minima closest to the given anchor points
def getMins(wns, spectrum, blPts):
    invSpec = np.multiply(spectrum, -1)
    specMinsAllInd, _ = find_peaks(invSpec)
    minsInd = []
    for point in blPts:
        tempMinInds = min(specMinsAllInd, key=lambda x:abs(point-wns[x]))
        minsInd.append(tempMinInds)
    return minsInd


#polynomial function, used by curve_fit
#indefinite number of args so that power can change fluidly 
def polynomialFit(x, *coeffs):
     y = np.polyval(coeffs, x)
     return y
    
#creates a baseline from the minima identified in getMins, making a straight line between adjacent points. 
def makeBaseline(minsInd, wns, spectrum): 
    baselineYVal = min(spectrum[minsInd])
    spectrum = np.subtract(spectrum, baselineYVal) 
    blYVals = spectrum[minsInd].tolist()
    #interpolate end baseline point
    m = (spectrum[minsInd[0]]-spectrum[minsInd[1]])/(minsInd[0]-minsInd[1])    
    endBl = (-1)*m*(minsInd[0])+spectrum[minsInd[0]]
    #add new end bl point to sets
    minsInd.insert(0,0)
    blYVals.insert(0, endBl)

    baseline = np.zeros(spectrum.shape)
    
    
    
    for i in range(len(minsInd)-1):
        interpX = [minsInd[i], minsInd[i+1]]
        interpY = [blYVals[i], blYVals[i+1]]
        miniBaseX = np.linspace(minsInd[i], minsInd[i+1], num=minsInd[i+1]-minsInd[i]+1)
        miniBaseY = np.interp(miniBaseX, interpX, interpY)
        baseline[minsInd[i]:minsInd[i+1]+1] = miniBaseY[:]
    
    return spectrum, baseline, minsInd, blYVals


def fitBaseline(minsInd, wns, spectrum): 
    baselineYVal = min(spectrum[minsInd])
    spectrum = np.subtract(spectrum, baselineYVal) 
    blYVals = spectrum[minsInd].tolist()
    #interpolate end baseline point
    m = (spectrum[minsInd[0]]-spectrum[minsInd[1]])/(minsInd[0]-minsInd[1])    
    endBl = (-1)*m*(minsInd[0])+spectrum[minsInd[0]]
    
    
    #add new end bl point to sets
    minsInd.insert(0,0)
    blYVals.insert(0, endBl)
    
    p0=np.ones(4)
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
wavenumbers, baselinePoints = getWnsandBL()
for chFolder in rawFTIRFolder.iterdir():
    dateFolder = chFolder / dateFolderNm
    chamberOutFolder = parentDir / folderNm / "1_baseline-corrected" / str(str(chFolder).split('\\')[-1:][0])
    dateOutFolder = chamberOutFolder / str(str(str(dateFolder).split('\\')[-1:][0]))
    dateOutFolder.mkdir(parents=True, exist_ok=True)
    replicateSets = getSets(dateFolder)
    for fileSet in replicateSets:
        dataSet = np.array([])
        for filename in fileSet:
           print(filename)
           specCounter+=1
            #print(filename)
           rawSpectrum = getSpec(dateFolder, filename)
           specMinsInd = getMins(wavenumbers, rawSpectrum, baselinePoints)
           #shiftSpec, blSpec, blXInds, blYIntVals = makeBaseline(specMinsInd, wavenumbers, rawSpectrum)
           shiftSpec, blSpec, blXInds, blYIntVals = fitBaseline(specMinsInd, wavenumbers, rawSpectrum)
           
           blCorrSpec = shiftSpec - blSpec
           plotBaseline(wavenumbers, shiftSpec, blSpec, blCorrSpec, blXInds, blYIntVals, filename)
           if np.any(dataSet)==False:
                 dataSet = blCorrSpec
           else:
                 dataSet = np.column_stack((dataSet, blCorrSpec))
                
                
        outputFile = np.column_stack((wavenumbers, dataSet))
        outputPath = dateOutFolder / str(str(filename[:-6].replace(" ", ""))+"_blCorr.csv")
        np.savetxt(outputPath, outputFile, '%5.7f', delimiter=',')
        
print(specCounter)