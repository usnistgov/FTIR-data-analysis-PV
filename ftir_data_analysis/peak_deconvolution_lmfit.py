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
https://lmfit.github.io/lmfit-py/constraints.html


For now, it uses a gaussian function I've defined in the code that allowed for the parameters wavenumber, area, 
fwhm, and c (y intercept). 
Implemented a parameter rat that represents the ratio of the peak area to height. This has been more successful than a 
height to width ratio at controlling peak shape, with more managageable values as well. I was hoping this would solve most of my problems with 
fitting, but I still sometimes get a peak that should be fitting smaller fitting as the larger peak in a group, and I may need to hard-code a parameter 
to handle this. I wish I could maintain more flexibility than that, but it is what it is for right now. It shouldn't be too hard to implement in the short term.
 Lmfit allows for this, where curve_fit from scipy.signal only allowed bounds. 
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
import re                   #regular expressions for list comprehension and string parsing 
import time                 #just used to monitor how long fits take to run

np.set_printoptions(suppress=True)

def getPosition(file):
    return int(re.sub('[a-zA-Z]+', '', file.split('-')[2]))

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
        pkPrmsTbl = np.genfromtxt(pkPrmFilePath, skip_header=1, delimiter=",")    #skips headers
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
def peakParameters(start, stop, maxh, paramsDir, nmWn):
    #import csv file 
    pkPrmFilePath = paramsDir / ("peak-params_" + start + "-" + stop + "-N" + str(nmWn) + "-lmfit.csv")
    pkPrmsTbl = np.genfromtxt(pkPrmFilePath, skip_header=1, delimiter=",")    #skips headers
    np.nan_to_num(pkPrmsTbl, nan=np.nan)
    numPeaks = len(pkPrmsTbl[:,0])
    
    fitParams = Parameters()        #creates Parameter object, part of lmfit. dict type  

    def miniParamGen(suffix, ind, varyBool):
        fitParams.add(name = f"p{pkID}_{suffix}", vary=varyBool, 
                          value=pkPrmsTbl[i, ind] if np.isnan(pkPrmsTbl[i,ind]) == False else None, 
                          min=pkPrmsTbl[i, ind-1] if np.isnan(pkPrmsTbl[i, ind-1]) == False else -np.inf, 
                          max=pkPrmsTbl[i, ind+1] if np.isnan(pkPrmsTbl[i, ind+1]) == False else np.inf)
    # save ratio constraints separately to refer to in the residual. 
    # lmfit prioritizes expressions over min and max of parameters in dependent parameters, and 
    # entering a min and max may cause clipping before the penalty can take effect.  
    ratConst = pkPrmsTbl[:, -3:]

    for i in range(numPeaks):       #currently hard-coded for gaussian function 
        pkID = i+1
        #add base parameters to fitParams first 
        miniParamGen('wn', 1, True), miniParamGen('area', 4, True), miniParamGen('fwhm', 7, True), miniParamGen('c', 10, False)
        #bug fix for area = 0 causing height = 0 causing zero division error in expression params
        if fitParams[f"p{pkID}_area"].min==0:
            fitParams[f"p{pkID}_area"].set(min=0.0001) 

        #add derived parameters 
        fitParams.add(f"p{pkID}_h", vary=False)
        fitParams[f"p{pkID}_h"].expr = f"p{pkID}_area / ( (p{pkID}_fwhm / (2*sqrt(2*log(2)))) *sqrt(2*pi))"
        
        fitParams.add(f"p{pkID}_rat", vary=False)
        fitParams[f"p{pkID}_rat"].expr = f"p{pkID}_area / p{pkID}_h"



        #come back here for custom relative parameters 
        # if start + "-" + stop == "1517-1900":
        #     print('add custom params here')

    return fitParams, numPeaks, ratConst

#custom gaussian function, though lmfit has built in models for gaussian, lorentzian, voight, etc that could be implemented if desired. 
def gauss_Area(x, wn, area, fwhm, c):
    return c + area/(fwhm*math.sqrt(math.pi/(4*math.log(2)))) * np.exp(-4*math.log(2)*(x-wn)**2/(fwhm**2))

def fitSection(start, stop, xSec, ySec, params, ratConst, numPeaks): 
    #local function for minimizer 
    def residual(pars, x, data, numPeaks): #lmfit requires set number of parameters for this function type to be passed into minimizer 
        #building model
        model = np.zeros_like(x)
        total_penalty = []
        
        # Force refresh of all derived values
        _ = pars.valuesdict()

        penalty_scale = np.max(data) * 2
        
        for i in range(1, numPeaks + 1):
            wn, area, fwhm, c = pars[f'p{i}_wn'], pars[f'p{i}_area'], pars[f'p{i}_fwhm'], pars[f'p{i}_c']
            
            # penalty handling - implemented using modifications of suggestions from Gemini 
            # penalties to prevent dependent parameters being ignored 
            # gemini so kindly told me at the end when i asked for more information about the penalty parameter 
            # that it's not necessarily the best way to do it and suggested I try soft contraints via expr. 
            # I'm not sure it will work for my almost-circular constraints, and I also don't want to do it right now.  

            rat_min, rat_max = ratConst[i-1, 0], ratConst[i-1, 2]   #get min and max from array pulled from csv 
            rat_val = r_val = pars[f'p{i}_rat'].value               #value being held in parameter at current iteration
            
        
            # Apply the penalty if it exceeds the max or min
            # use a squared penalty to keep the gradient smooth for the optimizer
            # penalty causes the residual to get huge when the expression parameter bounds are exceeded 
            if np.isfinite(rat_max):   
                total_penalty.append(np.maximum(0, rat_val - rat_max)**2 * penalty_scale)   #using np.maximum reduces number of if statements in loop and leaves penalties "vectorized" according to gemini which is supposed to be faster
            if np.isfinite(rat_min):
                total_penalty.append(np.maximum(0, rat_min - rat_val)**2 * penalty_scale)
            #add peak to model
            model += gauss_Area(x, wn, area, fwhm, c)

        # # Debug Print every 50 iterations
        # if iter_count % 50 == 0:
        #     print(f"Iter {iter_count} | Area[0]: {pars['p1_area'].value:.2f} | Rat[0]: {pars['p1_rat'].value:.2f} | Penalty: {penalty:.2e}")

        return np.concatenate([(model - data), total_penalty])
    
    mini = Minimizer(residual, params, fcn_args=(xSec, ySec, numPeaks))
    out = mini.minimize(method='leastsq')     #leastsq: Levenberg-Marquardt (default)
    
    # FIX: Slice the residual to match ySec's length
    # This ignores the penalty values at the end of the array
    clean_residual = out.residual[:len(ySec)]
    
    bestFit = ySec + clean_residual
    resultParams = out.params 
    return resultParams, bestFit

#convert lmfit output params to human-readable table for csv export. coded for gaussian more or less. 
def paramsToArray(params, xSec):
    wns, areas, fwhms, cs, heights = [], [], [], [], []
    for param in params.valuesdict():
        paramType, pkID = param.split("_")[1], param.split("_")[0].replace("p", "")
        # print(f"paramType: {paramType}, peakID: {pkID}")
        match paramType:
            case 'wn': wns.append(params.valuesdict()[param])
            case 'area': areas.append(params.valuesdict()[param])
            case 'fwhm': fwhms.append(params.valuesdict()[param])
            case 'c': cs.append(params.valuesdict()[param])
            case 'h': heights.append(params.valuesdict()[param])
    parsArr = np.array([wns, areas, fwhms, cs, heights]).T
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

def plotFitCheck(paramsFit, bestFit, xSec, ySec, file, colorList, numPeaks, rep):
    # numPeaks = int(len(paramsFit)/4)        #coded for gaussian 
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
        h = paramsFit.valuesdict()[f'p{pkID}_h']   
        rat = paramsFit.valuesdict()[f'p{pkID}_rat']   
        #apply parameters to gassian to plot individual peak
        onePkFit = gauss_Area(xSec, wn, area, fwhm, c)
        specXFit.plot(xSec, onePkFit, c=colorList[i], linewidth=0.5)
        #adding values to figure
        plt.figtext(0.1, (-0.05)*i, r'peak ' +str(pkID) +": " +str(round(wn, 2)) +'  area: ' + str(round(area,2)) +' FWHM: ' + str(round(fwhm, 2))+'  c: ' + str(round(c,3)) + '  h: ' + str(round(h,4)) + '  ratio: ' + str(round(rat,4)))
    specXFit.legend()
    return figSpecFit
    
#ACTUAL PROCESS
def fitOneFolder(
        directory, 
        rawDataFolder, 
        blFileNm='PET-baseline-wns-fit.txt',
        normDataFolderNm = "2_normalized",
        fitOutFolderNm = "3_fit-results",
        fitFigFolderNm = "3a_fit-results-figs",
        startStopList = [['565', '742'],['1517', '1900']],
        normWn = "723",
        totalFiles = 0, 
        fileCounter = 0
        ):
    
    folderStartTime = time.perf_counter()   #just for tracking how long it takes to run

    sourceFolder = Path(directory)
    rawDataFolder = Path(rawDataFolder)
    
    parentDir = rawDataFolder.absolute().parent 
    normDataFolder = parentDir / normDataFolderNm
    fitOutFolder = parentDir / fitOutFolderNm
    fitFigFolder = parentDir / fitFigFolderNm
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
        outputFigFolder = Path(root.replace(str(normDataFolder), str(fitFigFolder)))
        outputFolder.mkdir(parents=True, exist_ok=True)    
        outputFigFolder.mkdir(parents=True, exist_ok=True)   

        #looping through files in target folder 
        for filename in files: 
            if filename.endswith('Avg.csv')==False and 'Back' not in filename:                            #####ONLY DOING POS 1 for TESTING 
                
            #if filename=="20250130_PET-Ch1-Pos2-81h-Air_N723.csv":    #single file for testing 

                wavenumbers, dataSet = getSpecSet(filename, sourceFolder)
                yZeros = getMins(wavenumbers, dataSet[:,1], baselinePoints)
                xAll = wavenumbers
            
                fileCounter+=1
            
                for i in range(np.size(dataSet, 1)):
                    yAll = dataSet[:,i]
                    print(f'\r processing {str(fileCounter)} out of {str(totalFiles)} files (replicate {i+1} of {np.size(dataSet, 1)}), {str(round(((fileCounter-1)/totalFiles *100), 1))}% complete', end=' ')    #prints on one line
                    #loops through defined sections - sectioning is to enable easier fitting. sections are based on baseline points. 
                    for j in range(len(startStopList)):
                        xStartWn = startStopList[j][0]
                        xStopWn = startStopList[j][1]
                        startIndex, stopIndex, xSection, ySection = getSection(wavenumbers, yAll, yZeros, xStartWn, xStopWn)
                        maxHeight = max(ySection)
                        sectionParams, numberPeaks, ratioConstraints = peakParameters(xStartWn, xStopWn, maxHeight, parentDir, normWn) #gets parameters from params file
                        
                        outParams, sectionFit = fitSection(xStartWn, xStopWn, xSection, ySection, sectionParams, ratioConstraints, numberPeaks)
                        resultsArr = paramsToArray(outParams, xSection)
                        if j!=0: #this is just to skip plotting the first setion, 565-742, in the preview window, since this section's fit is relatively straightforward. 
    #                     #plotFit(initGuess_multi, popt_multi, xSection, ySection, filename)
                            fitFig = plotFitCheck(outParams, sectionFit, xSection, ySection, filename, plotColors, numberPeaks, i+1)
                        if j == 0:  #create the array for the fit results (for the first section)
                            allParams = resultsArr
                        else:       #add to the array for the fit results (for all following sections)
                            allParams = np.vstack([allParams, resultsArr])
                 
                    columnsList = ['wavenumbers', 'area (N'+normWn+')', 'FWHM', 'y int','height (N'+normWn+')']   
                
                    #dividing by all areas and adding as new columns
                    totalPeaks = allParams.shape[0]
                    for ind in range(totalPeaks):
                        allParams, columnsList = divByPeak(allParams, ind, columnsList, peakDict)

                    #adding column headers, making into pandas dataframe for export
                    outDf = pd.DataFrame(allParams, columns=columnsList)
                
                    outFilename = filename[:-4].replace(' ', '') + f"-fit_{i+1}.csv"
                    outFigFilename = outFilename.replace('.csv', '.png')
                    fitFig.savefig(outputFigFolder / outFigFilename, format='png', dpi=100, transparent=False, bbox_inches='tight')
                    outDf.to_csv(outputFolder / outFilename, index=False)

                fileCompleteTime = time.perf_counter()
                elapsedSeconds = fileCompleteTime - folderStartTime
                elapsedMinutes = elapsedSeconds / 60
                print(f'\r processing {str(fileCounter)} out of {str(totalFiles)} files (replicate {i+1} of {np.size(dataSet, 1)}), {str(round(((fileCounter)/totalFiles *100), 1))}% complete, {round(elapsedMinutes, 2)} minutes elapsed', end=' ')    #prints on one line
    print("Program finished!\\a") 
    return fileCounter

def fitMulti(normDataFolder, rawDataFolder, startExp, stopExp):
    totalFolders=0
    totalFiles=0
    for root, dirs, files in os.walk(normDataFolder): 
        if len(files)>0 and startExp <= int(root.split('//')[-1].split('-')[-1].replace('h', '')) <= stopExp:
            totalFolders+=1
            totalFiles += len(files)
    totalFiles = totalFiles/2  # removing averages 
    print(f"total files in batch (not counting replicates): {totalFiles}")
    
    currentFolder=0
    for root, dirs, files in os.walk(normDataFolder): 
        if len(files)>0 and startExp <= int(root.split('//')[-1].split('-')[-1].replace('h', '')) <= stopExp:
            print(root)
            currentFolder+=1
            print(f"Beginning folder {currentFolder} of {totalFolders}")
            fitOneFolder(root, rawDataFolder)

    
#fitOneFolder("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-exposure-ND-filters/2_normalized/723/chamber-1/20250205-104h", "C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-exposure-ND-filters/0_raw-data")
# fitMulti("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr/2_normalized/723", "C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr/0_raw-data", 226, 226)

## !! Redo up to 226 hrs to correct bad lower bound on 1745 ratio 
## done: 277, 327, 400, 564, 605, 694, 759, 871, 1095, 1171, 1290, 1393
fitMulti("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr/2_normalized/723", "C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr/0_raw-data", 1681, 1681)
fitMulti("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr/2_normalized/723", "C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr/0_raw-data", 2107, 2107)

# ###REDOS
# fitMulti("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr/2_normalized/723", "C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr/0_raw-data", 0, 0)
# fitMulti("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr/2_normalized/723", "C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr/0_raw-data", 12, 12)
# fitMulti("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr/2_normalized/723", "C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr/0_raw-data", 25, 25)
# fitMulti("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr/2_normalized/723", "C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr/0_raw-data", 45, 45)
# fitMulti("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr/2_normalized/723", "C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr/0_raw-data", 63, 63)
# fitMulti("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr/2_normalized/723", "C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr/0_raw-data", 81, 81)
# fitMulti("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr/2_normalized/723", "C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr/0_raw-data", 104, 104)
# fitMulti("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr/2_normalized/723", "C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr/0_raw-data", 145, 145)
# fitMulti("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr/2_normalized/723", "C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr/0_raw-data", 185, 185)
# fitMulti("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr/2_normalized/723", "C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr/0_raw-data", 226, 226)

#fitMulti("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr/2_normalized/723", "C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr/0_raw-data", 605, 605)
# fitOneFolder("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr/2_normalized/723/chamber-5/20250110-0h", "C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr/0_raw-data")
#fitOneFolder("//Cfs2e.nist.gov/73_el/731/internal/CONFOCAL/FS2/Data4/Hsiuchin/reciprocity experiment (3M PET)/FTIR/cutoff 305/KLJ-processing/2_normalized/723/16h", "//cfs2e.nist.gov/73_EL/731/internal/CONFOCAL/FS2/Data4/Hsiuchin/reciprocity experiment (3M PET)/FTIR/cutoff 305/KLJ-processing/0_raw-data")