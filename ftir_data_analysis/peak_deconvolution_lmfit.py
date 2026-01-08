# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 14:08:22 2025

@author: klj

performs deconvolution on spectrum sections 565 - 742 (current normalization peak region) and 1517-1900
(carbonyl region). combines results for both sections and outputs fit peak pararmeters as new csv. 
currently adds columns with the result of each peak area divided by every other peak area for further ratio analysis. 
saves peak parameters in csv

There is lots of room to improve the fitting function with a more in depth exploration of lmfit's capabilities. 
This is something that can be added with time. See:
https://lmfit.github.io/lmfit-py/builtin_models.html#example-3-fitting-multiple-peaks-and-using-prefixes

For now, it uses a gaussian function I've defined in the code that allowed for the parameters wavenumber, area, 
fwhm, and c (y intercept). 
The next step is to implement a parameter for height to width ratio that allows for constraint of 
relative peak dimensions. Lmfit allows for this, where curve_fit from scipy.signal only allowed bounds. 
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks as find_peaks
from lmfit import Minimizer, Parameters, Parameter, Model #create_params, report_fit
import matplotlib.pyplot as plt
import math
import colorsys

np.set_printoptions(suppress=True)

#grabs group of 4 spectra and splits into wavenumber and data columns
#input: fileNm (file name of spectra set, output by pre-processing scripts)
#       directory (raw data folder, will automatically search for associated processed files)
#output: wns (single-column numpy array with wavenumbers)
#        allSpec (multi-column numpy array, each column one replicate spectrum, no wavenumber column)
def getSpecSet(fileNm, directory):
    file = np.loadtxt(directory / fileNm, delimiter=',')
    wns = file[:,0]
    allSpec = np.delete(file, 0, 1)
    return wns, allSpec

#gets spectral minima to use as spot to separate into fitting sections
def getMins(wns, spectrum, blPts):
    invSpec = np.multiply(spectrum, -1)
    specMinsAllInd, _ = find_peaks(invSpec)
    minsInd = []
    for point in blPts:
        tempMinInds = min(specMinsAllInd, key=lambda x:abs(point-wns[x]))
        minsInd.append(tempMinInds)
    return minsInd

#returns section of spectrum 
def getSection(wns, spectrum, zeros, startWn, stopWn):
    startInd = min(zeros, key=lambda x:abs(float(startWn)-wns[x]))
    if abs(wns[startInd] - float(startWn)) > 20:                                            #duct tape error handling for when the boundaries needed don't fit with the best baseline anchors when fitting a baseline
        startInd = np.where(wns == (min(wns, key=lambda x: abs(float(startWn)-x))))[0][0]
    stopInd = min(zeros, key=lambda x:abs(float(stopWn)-wns[x]))
    if abs(wns[stopInd] - float(stopWn)) > 20:                                              #duct tape error handling for when the boundaries needed don't fit with the best baseline anchors when fitting a baseline
        stopInd = np.where(wns == (min(wns, key=lambda x: abs(float(stopWn)-x))))[0][0]
    xSubsection = wns[startInd:stopInd+1]
    ySubsection = spectrum[startInd:stopInd+1]
    return startInd, stopInd, xSubsection, ySubsection

def createPeakDict(ssList, paramsDir, nWn):
    peakDictionary = {}
    ind=0
    for j in range(len(ssList)):
        startWn, stopWn = ssList[j][0], ssList[j][1]
        pkPrmFilePath = paramsDir / ("peak-params_" + startWn + "-" + stopWn + "-N" + str(nWn) + "-lmfit.csv")
        pkPrmsTbl = np.loadtxt(pkPrmFilePath, skiprows=1, delimiter=",")    #skips headers
        for wn in pkPrmsTbl[:,1]:
            peakDictionary[ind] = str(int(wn))
            ind+=1       
    return peakDictionary

#returns list of evenly-spaced colors to use for plot 
def genPlotColors(pkDict):
    totPks = len(pkDict)
    hlsTups = [(x/totPks, 0.4, 1) for x in range(totPks)]
    rgbTups = list(map(lambda x: colorsys.hls_to_rgb(*x), hlsTups))
    return rgbTups

#creates peak parameters for fitting using lmfit. 
def peakParameters(start, stop, paramsDir, nmWn):
    #import csv file 
    pkPrmFilePath = paramsDir / ("peak-params_" + start + "-" + stop + "-N" + str(nmWn) + "-lmfit.csv")
    pkPrmsTbl = np.loadtxt(pkPrmFilePath, skiprows=1, delimiter=",")
    numPeaks = len(pkPrmsTbl[:,0])

    fitParams = Parameters()        #creates Parameter object, part of lmfit. dict type  
    for i in range(numPeaks):       #currently hard-coded for gaussian function 
        pkID = i+1
        wn = Parameter(name = f"p{pkID}_wn", value=pkPrmsTbl[:,1][i], vary=True, min=pkPrmsTbl[:,0][i], max=pkPrmsTbl[:,2][i])
        area = Parameter(name = f"p{pkID}_area", value=pkPrmsTbl[:,4][i], vary=True, min=pkPrmsTbl[:,3][i], max=pkPrmsTbl[:,5][i])
        fwhm = Parameter(name = f"p{pkID}_fwhm", value=pkPrmsTbl[:,7][i], vary=True, min=pkPrmsTbl[:,6][i], max=pkPrmsTbl[:,8][i])
        c = Parameter(name = f"p{pkID}_c", value=pkPrmsTbl[:,10][i], vary=False)
        fitParams.add_many((wn), (area), (fwhm), (c))

    return fitParams, numPeaks

#custom gaussian function, though lmfit has built in models for gaussian, lorentzian, voight, etc that could be implemented if desired. 
def gauss_Area(x, wn, area, fwhm, c):
    return c + area/(fwhm*math.sqrt(math.pi/(4*math.log(2)))) * np.exp(-4*math.log(2)*(x-wn)**2/(fwhm**2))

#not currently implemented! requires some adjusting to use csv input parameters. 
# def gauss_Height(x, wn, amp, fwhm, c):
#     return c + amp * np.exp(-0.5 * (((x-wn)**2)/(fwhm**2)))

def residual(pars, x, data): #lmfit requires set number of parameters for this function type to be passed into minimizer 
    numPks = int(len(pars)/4) #hard coded for 4 coeffs 
    model = gauss_Area(x, pars['p1_wn'], pars['p1_area'], pars['p1_fwhm'], pars['p1_c'])
    #model = Model(gauss_Area, prefix='p1_')
    for i in range(1, numPks):
        pkID = i+1
        model = model + gauss_Area(x, pars[f'p{pkID}_wn'], pars[f'p{pkID}_area'], pars[f'p{pkID}_fwhm'], pars[f'p{pkID}_c'])
    return model - data 

def fitSection(start, stop, xSec, ySec, params, numPeaks): 
    mini = Minimizer(residual, params, fcn_args=(xSec, ySec))
    out = mini.minimize(method='leastsq')     #leastsq: Levenberg-Marquardt (default)
    bestFit = ySec + out.residual 
    resultParams = out.params
    return resultParams, bestFit

#convert lmfit output params to human-readable table for csv export 
def paramsToArray(params):
    wns, areas, fwhms, cs = [], [], [], []
    for param in params.valuesdict():
        paramType, pkID = param.split("_")[1], param.split("_")[0].replace("p", "")
        #print(f"paramType: {paramType}, peakID: {pkID}")
        match paramType:
            case 'wn': wns.append(params.valuesdict()[param])
            case 'area': areas.append(params.valuesdict()[param])
            case 'fwhm': fwhms.append(params.valuesdict()[param])
            case 'c': cs.append(params.valuesdict()[param])
    parsArr = np.array([wns, areas, fwhms, cs]).T
    return parsArr

#input - array of resulting parameters to add normalized column to (paramsArr), index of the column
#containing the values for the normalization peak (normWnListInd), and list of column headers to use
#for input array. 
#outputs array with new area column added (area values for all peaks divided by areas for normalization peak),
#and new updated column header list with new column added. 
def divByPeak(paramsArr, pkIndex, colList, pkDict):
    wnList = paramsArr[:,0].tolist()            #create list of wns from array 
    divisorArea = paramsArr[:,1][pkIndex]  #col index 1: area 
    divAreaList = []
    #loops through peaks and divides absolute area by divisor peak area to create ratios. 
    for i in range(len(wnList)):
        divAreaList.append((paramsArr[:,1][i])/divisorArea)
    dividedArea = np.array([divAreaList])
    paramsArr = np.concatenate((paramsArr, dividedArea.T), axis=1)
    #adds column name for divided ratio peak to column name list, which is added just before saving.
    colList.append('divided Area (by'+str(pkDict[pkIndex])+')')
    return paramsArr, colList     

def plotFitCheck(paramsFit, bestFit, xSec, ySec, file, colorList, rep):
    numPeaks = int(len(paramsFit)/4)        #coded for gaussian 
    yFit = bestFit
    figSpecFit, specXFit = plt.subplots(figsize=(10,5))
    plt.title(f"{file} r{rep}")
    specXFit.plot(xSec, ySec, c='black', label='spectrum', linewidth=0.5)
    specXFit.plot(xSec, yFit, c='red', label='fit', linewidth=0.5)
    #loops through to plot individual peak functions
    for i in range(numPeaks):
        pkID = i+1
        #grabbing params from paramsFit results dict 
        wn = paramsFit.valuesdict()[f'p{pkID}_wn']    
        area = paramsFit.valuesdict()[f'p{pkID}_area']   
        fwhm = paramsFit.valuesdict()[f'p{pkID}_fwhm']   
        c = paramsFit.valuesdict()[f'p{pkID}_c']   
        #apply parameters to gassian to plot individual peak
        onePkFit = gauss_Area(xSec, wn, area, fwhm, c)
        specXFit.plot(xSec, onePkFit, c=colorList[i], linewidth=0.5)
        #adding values to figure
        plt.figtext(0.1, (-0.05)*i, r'peak ' +str(pkID) +": " +str(round(wn, 2)) +'  area: ' + str(round(area,2)) +' FWHM: ' + str(round(fwhm, 2))+'  c: ' + str(round(c,3)))
    specXFit.legend()
    
#ACTUAL PROCESS
def fitOneFolder(
        directory, 
        rawDataFolder, 
        blFileNm='PET-baseline-wns-fit.txt',
        normDataFolderNm = "2_normalized",
        fitOutFolderNm = "3_fit-results",
        startStopList = [['565', '742'],['1517', '1900']],
        normWn = "723",
        totalFiles = 0, 
        fileCounter = 0
        ):
    sourceFolder = Path(directory)
    rawDataFolder = Path(rawDataFolder)
    
    parentDir = rawDataFolder.absolute().parent 
    normDataFolder = parentDir / normDataFolderNm
    fitOutFolder = parentDir / fitOutFolderNm
    fitOutFolder.mkdir(parents=True, exist_ok=True)

    #get baseline anchor points file. 
    blFilePath = parentDir / blFileNm      #this is just the file with the starting anchor points. 
    baselinePoints = np.loadtxt(blFilePath, delimiter=" ")
    peakDict = createPeakDict(startStopList, parentDir, normWn)
    plotColors = genPlotColors(peakDict)
    
    #count files in folder 
    if totalFiles==0: 
        for root, dirs, files in os.walk(sourceFolder):
            for file in files:
                if file.endswith('Avg.csv')==False: 
                    totalFiles +=1
    print(f'total file sets for fitting: {totalFiles}')
    
    for root, dirs, files in os.walk(sourceFolder):
        depth = root.replace(str(directory), '').count(os.sep)
        outputFolder = Path(root.replace(str(normDataFolder), str(fitOutFolder)))
        outputFolder.mkdir(parents=True, exist_ok=True)    

        #looping through files in target folder 
        for filename in files: 
            if filename.endswith('Avg.csv')==False:
            #if filename=="20250130_PET-Ch1-Pos2-81h-Air_N723.csv":    #single file for testing 

                wavenumbers, dataSet = getSpecSet(filename, sourceFolder)
                yZeros = getMins(wavenumbers, dataSet[:,1], baselinePoints)
                xAll = wavenumbers
            
                fileCounter+=1
                print(f'\r processing {str(fileCounter)} out of {str(totalFiles)} files, {str(round(((fileCounter-1)/totalFiles *100), 1))}% complete', end=' ')    #prints on one line
            
                for i in range(np.size(dataSet, 1)):
                    yAll = dataSet[:,i]
                    #loops through defined sections - sectioning is to enable easier fitting. sections are based on baseline points. 
                    for j in range(len(startStopList)):
                        xStartWn = startStopList[j][0]
                        xStopWn = startStopList[j][1]
                        startIndex, stopIndex, xSection, ySection = getSection(wavenumbers, yAll, yZeros, xStartWn, xStopWn)
                        sectionParams, numberPeaks = peakParameters(xStartWn, xStopWn, parentDir, normWn) #gets parameters from params file
                        
                        outParams, sectionFit = fitSection(xStartWn, xStopWn, xSection, ySection, sectionParams, numberPeaks)
                        resultsArr = paramsToArray(outParams)
    #                 if j!=0: #this is just to skip plotting the first setion, 565-742, in the preview window, since this section's fit is relatively straightforward. 
    #                     #plotFit(initGuess_multi, popt_multi, xSection, ySection, filename)
                        plotFitCheck(outParams, sectionFit, xSection, ySection, filename, plotColors, i+1)
                        if j == 0:  #create the array for the fit results (for the first section)
                            allParams = resultsArr
                        else:       #add to the array for the fit results (for all following sections)
                            allParams = np.vstack([allParams, resultsArr])
                 
                    columnsList = ['wavenumbers', 'area (N'+normWn+')', 'FWHM', 'y int'] #,'height (N'+normWn+')']   
                
                    #dividing by all areas and adding as new columns
                    totalPeaks = allParams.shape[0]
                    for ind in range(totalPeaks):
                        allParams, columnsList = divByPeak(allParams, ind, columnsList, peakDict)

                    #adding column headers, making into pandas dataframe for export
                    outDf = pd.DataFrame(allParams, columns=columnsList)
                
                    outFilename = filename[:-4].replace(' ', '') + f"-fit_{i+1}.csv"
                    outDf.to_csv(outputFolder / outFilename, index=False)

                print(f'\r processing {str(fileCounter)} out of {str(totalFiles)} files, {str(round((fileCounter/totalFiles *100), 1))}% complete', end=' ')    #prints on one line
    return fileCounter


    
fitOneFolder("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-exposure/2_normalized/723/chamber-5/20251125-2107h", "C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-exposure/0_raw-data")

#fitOneFolder("//Cfs2e.nist.gov/73_el/731/internal/CONFOCAL/FS2/Data4/Hsiuchin/reciprocity experiment (3M PET)/FTIR/cutoff 305/KLJ-processing/2_normalized/723/16h", "//cfs2e.nist.gov/73_EL/731/internal/CONFOCAL/FS2/Data4/Hsiuchin/reciprocity experiment (3M PET)/FTIR/cutoff 305/KLJ-processing/0_raw-data")