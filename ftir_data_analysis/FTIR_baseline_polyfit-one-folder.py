# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 13:52:09 2025

@author: klj
"""
#uses a baseline anchor points file (txt) to automate baseline
#correction for bulk FTIR data. Works by finding minima in spectra and finding a set of 
#minima closest to the baseline anchor point starting values. The baseline created is 
#a polynomial fit. Remember that some distorion of peaks will be inevitable with any baseline
#correction.

#file names for raw data are critical DOWNSTREAM as the script parses the file name as a string for
#naming output files as well as finding corresponding doses downstream. 
#file names must be of the following format (to use as written with no changes): 
    #20250110_PET-Ch1-Pos1-0h-Air_1.CSV
    #YYYYMMDD_identity-ChX-PosY-Zh-SideID_R.CSV
    #YYYYMMDD is year-month-day
    #identity is material abbreviation (such as PET)
    #X is chamber number 
    #Y is position (as in 1 of 17 positions in holder)
    #Z is exposure hours (no decimals - match this to "File suffix" in doses file, 
    #Doses-Ch-1.CSV or Doses-Ch-5.CSV for 2025 PET exposure)
    #SideID should be "Air" or "Back" exactly 
    #R is replicate number, likely 1-4
    
#this puts replicates together into one csv with multiple columns (one wavenumber column and x replicate columns)    
    
#outputs to another folder in the same directory as the raw data folder, called 1_baseline-corrected
#currently, working on data in C:\Users\klj\OneDrive - NIST\Projects\PV-Project\Reciprocity\FTIR-data-PET-exposure

#This currently requires all data to be the same spacing and range as each other, but not 
#necessarily the same as the data it was written for. 

import os 
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks as find_peaks
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#__all__ = ["baselineLoop"]


# folderNm='FTIR-data-PET-exposure'  #name of folder where FTIR data is stored - this will contain all downstream outputs as well in individual subfolders
# parentDir = Path().absolute().parent
# blFilePath = parentDir / folderNm / 'PET-baseline-wns-fit.txt'      #this is just the file with the starting anchor points. 
# rawFTIRFolder = parentDir / folderNm / "0_raw-data"             #all raw data (CSV only) placed here 
# wnMidFolder = "chamber-1"                                       #this needs to be one of the chambers being used, but won't need to be changed, it's only for obtaining the wn column.


#obtains the wavenumber column and baseline file data using the folders and files named above. 
def getWnsandBL(rawFolder, wnFolder, blPath):
    wnFile = os.listdir(rawFolder / wnFolder / str((os.listdir(rawFolder / wnFolder)[0])))[0]
    wnFileLoc = (rawFolder / wnFolder / str((os.listdir(rawFolder / wnFolder)[0]))) / wnFile
    wns = np.loadtxt(wnFileLoc, delimiter=',')[:,0]
    blPoints = np.loadtxt(blPath, delimiter=" ")
    print(blPoints)
    return wns , blPoints

#makes a list of all replicate file names for each sample measurement. 
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


def fitBaseline(minsInd, wns, spectrum, pOrder): 
    baselineYVal = min(spectrum[minsInd])
    spectrum = np.subtract(spectrum, baselineYVal) 
    blYVals = spectrum[minsInd].tolist()
    #interpolate end baseline point
    m = (spectrum[minsInd[0]]-spectrum[minsInd[1]])/(minsInd[0]-minsInd[1])    
    endBl = (-1)*m*(minsInd[0])+spectrum[minsInd[0]]
    
    
    #add new end bl point to sets
    minsInd.insert(0,0)
    blYVals.insert(0, endBl)
    
    p0=np.ones(pOrder+1)
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
        directory,
        folderNm='FTIR-data-PET-exposure',
        polyOrder=3,
        blFileNm='PET-baseline-wns-fit.txt',
        rawFTIRFileNm='0_raw-data',
        wnMidFolder="chamber-1"
        ):
    np.set_printoptions(suppress=True)
    specCounter = 0
    
    #folderNm='FTIR-data-PET-exposure'  #name of folder where FTIR data is stored - this will contain all downstream outputs as well in individual subfolders
    #parentDir = Path().absolute().parent
    parentDir=Path(directory)
    blFilePath = parentDir / folderNm / blFileNm      #this is just the file with the starting anchor points. 
    rawFTIRFolder = parentDir / folderNm / rawFTIRFileNm             #all raw data (CSV only) placed here 
                                          #this needs to be one of the chambers being used, but won't need to be changed, it's only for obtaining the wn column. 
    wavenumbers, baselinePoints = getWnsandBL(rawFTIRFolder, wnMidFolder, blFilePath)
    for chFolder in rawFTIRFolder.iterdir():
        for dateFolder in chFolder.iterdir():
    
            #making destination folders 
            chamberOutFolder = parentDir / folderNm / "1_baseline-corrected" / str(str(chFolder).split('\\')[-1:][0])
            dateOutFolder = chamberOutFolder / str(str(str(dateFolder).split('\\')[-1:][0]))
            dateOutFolder.mkdir(parents=True, exist_ok=True)
            replicateSets = getSets(dateFolder)
            
            for fileSet in replicateSets:
                dataSet = np.array([])
                for filename in fileSet:
                    #print(filename)
                    specCounter+=1
                    #print(filename)
                    rawSpectrum = getSpec(dateFolder, filename)
                    specMinsInd = getMins(wavenumbers, rawSpectrum, baselinePoints)
                    #shiftSpec, blSpec, blXInds, blYIntVals = makeBaseline(specMinsInd, wavenumbers, rawSpectrum)
                    shiftSpec, blSpec, blXInds, blYIntVals = fitBaseline(specMinsInd, wavenumbers, rawSpectrum, polyOrder)
                   
                    blCorrSpec = shiftSpec - blSpec
                    plotBaseline(wavenumbers, shiftSpec, blSpec, blCorrSpec, blXInds, blYIntVals, filename)
                    if np.any(dataSet)==False:
                          dataSet = blCorrSpec
                    else:
                          dataSet = np.column_stack((dataSet, blCorrSpec))
                        
                        
                outputFile = np.column_stack((wavenumbers, dataSet))
                outputPath = dateOutFolder / str(str(filename[:-6].replace(" ", ""))+"_blCorr.csv")
                np.savetxt(outputPath, outputFile, '%5.7f', delimiter=',')
            
    print(f"total files baseline corrected: {specCounter}")

#baselineLoop()

if __name__ == "__main__":          #does not run if importing only if running
    baselinePolyFitLoop("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity")

#Path().absolute().parent