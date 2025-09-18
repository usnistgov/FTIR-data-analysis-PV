# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 16:50:27 2025

@author: klj
"""

##performs deconvolution on spectrum sections 565 - 742 (current normalization peak region) and 1517-1900
##(carbonyl region). combines results for both sections and outputs fit peak pararmeters as new csv. 
##currently adds columns with the result of each peak area divided by every other peak area for further ratio analysis. 
##saves peak parameters in csv


#import os 
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks as find_peaks
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math

np.set_printoptions(suppress=True)

#grabs group of 4 spectra and splits into wavenumber and data columns
def getSpecSet(fileNm, directory):
    file = np.loadtxt(directory / fileNm, delimiter=',')
    wns = file[:,0]
    allSpec = np.delete(file, 0, 1)
    return wns, allSpec

#ets spectral minima to use as spot to separate into fitting sections
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
    stopInd = min(zeros, key=lambda x:abs(float(stopWn)-wns[x]))
    xSubsection = wns[startInd:stopInd+1]
    ySubsection = spectrum[startInd:stopInd+1]
    
    return startInd, stopInd, xSubsection, ySubsection

#gets peak fitting parameters from bounds file and puts into tuple format for curve_fit
def peakParameters(start, stop):
    pkPrmFilePath = parentDir / folderNm / ("peak-params_" + start + "-" + stop + "-N" + str(normWn) + ".csv")
    pkPrmsTbl = np.loadtxt(pkPrmFilePath, skiprows=1, delimiter=",")
    #print(pkPrmsTbl)
    initGuessList = []
    lowBoundList = []
    highBoundList = []
    for i in range(len(pkPrmsTbl[:,0])):
        initGuessList.append(pkPrmsTbl[:,1][i]) #wnInitGuess
        initGuessList.append(pkPrmsTbl[:,4][i]) #area guess
        initGuessList.append((pkPrmsTbl[:,7][i])) #fwhm initial, 
        initGuessList.append(pkPrmsTbl[:,10][i])
        lowBoundList.append(pkPrmsTbl[:,0][i]) #wn low bound
        lowBoundList.append(pkPrmsTbl[:,3][i]) #area low bound
        lowBoundList.append((pkPrmsTbl[:,6][i])) #fwhm low bound
        lowBoundList.append(pkPrmsTbl[:,9][i]) #c low bound
        highBoundList.append(pkPrmsTbl[:,2][i]) #wn hi bound
        highBoundList.append(pkPrmsTbl[:,5][i]) #area high bound
        highBoundList.append((pkPrmsTbl[:,8][i])) #fwhm hi bound 
        highBoundList.append(pkPrmsTbl[:,11][i]) # c hi bound
   
    boundsTuple = tuple(lowBoundList), tuple(highBoundList)
    return initGuessList, boundsTuple

def createPeakDict(ssList):
    peakDictionary = {}
    ind=0
    for j in range(len(ssList)):
        startWn, stopWn = ssList[j][0], ssList[j][1]
        pkPrmFilePath = parentDir / folderNm / ("peak-params_" + startWn + "-" + stopWn + "-N" + str(normWn) + ".csv")
        pkPrmsTbl = np.loadtxt(pkPrmFilePath, skiprows=1, delimiter=",")
        #print(pkPrmsTbl[:,1])
        for wn in pkPrmsTbl[:,1]:
            peakDictionary[ind] = str(int(wn))
            ind+=1
            
    return peakDictionary
    
#gaussian function, used by curve_fit
#indefinite number of args so that number of peaks can change fluidly 
def gauss_function_multi(x, *args):
    total_val = 0
    
    for i in range(int(len(args) / 4)):
        x0n   = args[i*4 + 0]
        An    = args[i*4 + 1]
        fwhmn = args[i*4 + 2]
        cn    = args[i*4 + 3]
        
        total_val += cn + An/(fwhmn*math.sqrt(math.pi/(4*math.log(2)))) * np.exp(-4*math.log(2)*(x-x0n)**2/(fwhmn**2))
    
    return total_val

def fitSection(start, stop, xSec, ySec):
    initGuess, bounds = peakParameters(start, stop) #gets parameters from params file
    paramsFit, pcov = curve_fit(gauss_function_multi, xSec, ySec, p0=initGuess, bounds=bounds)   #actual fit 
    numPeaks = int(len(initGuess)/4) 
    #finding heights 
    heightsList = []
    for i in range(numPeaks):
        yFitTemp = gauss_function_multi(xSec, *paramsFit[int(i*4):int(i*4+4)])
        peakHeight = max(yFitTemp)
        heightsList.append(peakHeight)   
    heights = np.array([heightsList])
    paramsFitArr = np.resize(paramsFit, (numPeaks,4))
    paramsFitArr = np.concatenate((paramsFitArr, heights.T), axis=1)
    return initGuess, bounds, paramsFit, pcov, paramsFitArr

#input - array of resulting parameters to add normalized column to (paramsArr), index of the column
#containing the values for the normalization peak (normWnListInd), and list of column headers to use
#for input array. 
#outputs array with new area column added (area values for all peaks divided by areas for normalization peak),
#and new updated column header list with new column added. 
def divByPeak(paramsArr, divWnListInd, colList):
    wnList = paramsArr[:,0].tolist()
    divisorArea = paramsArr[:,1][divWnListInd]
    divAreaList = []
    for i in range(len(wnList)):
        divAreaList.append((paramsArr[:,1][i])/divisorArea)
    dividedArea = np.array([divAreaList])
    paramsArr = np.concatenate((paramsArr, dividedArea.T), axis=1)
    colList.append('divided Area (by'+str(peakDict[divWnListInd])+')')
    return paramsArr, colList     

def plotFitCheck(initGuessList, paramsFit, xSec, ySec, file):
    yFit = gauss_function_multi(xSec, *paramsFit)
    figSpecFit, specXFit = plt.subplots(figsize=(10,5))
    plt.title(file)
    specXFit.plot(xSec, ySec, c='black', label='spectrum', linewidth=0.5)
    specXFit.plot(xSec, yFit, c='red', label='fit', linewidth=0.5)
    
    for i in range(int(len(initGuessList)/4)):
        colorList = 'green', 'orange', 'purple', 'pink', 'brown', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'blue'
        wnTemp = paramsFit[int(i*4)] 
        intTemp = paramsFit[int(i*4+1)]
        FWHMTemp = paramsFit[int(i*4+2)]
        cTemp = paramsFit[int(i*4+3)]
        yFitTemp = gauss_function_multi(xSec, *paramsFit[int(i*4):int(i*4+4)])
        specXFit.plot(xSec, yFitTemp, c=colorList[i], linewidth=0.5)
        plt.figtext(0.1, (-0.05)*i, r'peak ' +str(i) +": " +str(round(wnTemp, 2)) +'  area: ' + str(round(intTemp,2)) +' FWHM: ' + str(round(FWHMTemp, 2))+'  c: ' + str(round(cTemp,3)))
    specXFit.legend()
  
# #cleaned up, formatted plots for publication - higher quality and take longer to process.     
# def plotFit(initGuessList, paramsFit, xSec, ySec, file):
#     yFit = gauss_function_multi(xSec, *paramsFit)
#     figSpecFit, specXFit = plt.subplots(figsize=(8,6), dpi=100)
#     #set up plot formatting etc 
#     plt.title(file, fontsize=20, y=1.05, family = fontString)
#     specXFit.set_xlabel("wavenumbers (cm\u207b\u00B9)", fontsize=20, font = fontString)
#     specXFit.set_ylabel(f"Absorbance (normalized to {normWn})", fontsize=20, font = fontString)
#     specXFit.tick_params(axis='both', which='major', labelsize=18)
#     for tick in specXFit.get_xticklabels():
#         tick.set_fontname(fontString)
#     for tick in specXFit.get_yticklabels():
#         tick.set_fontname(fontString)
#     specXFit.axis([1855, 1525, -0.05, 0.95])
#     #plot spectrum and fit
#     specXFit.plot(xSec, ySec, c='gray', label='spectrum', linewidth=1.5)
#     specXFit.plot(xSec, yFit, c='pink', label='fit', linewidth=2, linestyle="--")
#     wnList = []
#     intList = []
#     for i in range(int(len(initGuessList)/4)):
#         colorList = 'green', 'orange', 'purple', 'pink', 'brown', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'blue'
#         wnTemp = paramsFit[int(i*4)] 
#         wnList.append(wnTemp)
#         yFitTemp = gauss_function_multi(xSec, *paramsFit[int(i*4):int(i*4+4)])
#         intList.append((yFitTemp[xSec.tolist().index(min(xSec, key=lambda x:abs(x-wnTemp)))]))
#         specXFit.plot(xSec, yFitTemp, c=colorList[i], linewidth=1.5)
#         specXFit.annotate(str(round(wnTemp, 1)), xy=(wnList[i], intList[i]+0.02), xytext=(wnList[i],intList[i]+0.1), arrowprops=dict(width=1, headwidth=4, headlength=3, color = colorList[i]), ha='center', rotation="vertical", fontsize=15, family = fontString)
#     specXFit.legend(fontsize=18, facecolor='none', edgecolor='none', prop=dict(family=fontString, size=18))
#     figOutPath = chamberFigOutFolder / dateFolder 
#     figOutPath.mkdir(parents=True, exist_ok=True)
#     figFile = file.split(".")[0] + "-deconvFig.png"
#     figSpecFit.savefig(figOutPath / figFile, format='png', dpi=100, transparent=True)


#ACTUAL PROCESS
def fitOneFile(chFolder, dateFolder, totalFiles, fileCounter):
    directory = normFTIRFolder / str(normWn) / chFolder / dateFolder  
    chamberOutFolder = parentDir / folderNm / "3_fit-results" / normWn / str(str(chFolder).split('\\')[-1:][0])
    chamberFigOutFolder = parentDir / folderNm / "x_deconv-figs" / normWn / str(str(chFolder).split('\\')[-1:][0])
    chamberOutFolder.mkdir(parents=True, exist_ok=True)         #creates folder if it doesn't exist 
    chamberFigOutFolder.mkdir(parents=True, exist_ok=True)      #creates folder if it doesn't exist 
    
    for file in directory.iterdir(): 
        
        filename = str(file).split('\\')[-1]
        pos = filename.split("-")[2]
        expHrs = filename.split('-')[3]
        skipCheckTuple = (str(chFolder).split("\\")[-1], expHrs) #any files that need to be skipped to fit manually should be added to this list 
        
        if filename[-7:] != "Avg.csv" and skipCheckTuple not in skipList and filename.split("-")[-1].split("_")[0]=="Air":
            #obtaining x and y data sets 
            wavenumbers, dataSet = getSpecSet(filename, directory)
            blPoints = np.loadtxt(blFilePath, delimiter=" ")
            yZeros = getMins(wavenumbers, dataSet[:,1], blPoints)
            xAll = wavenumbers
            
            fileCounter+=1
            print(f'\r processing {str(fileCounter)} out of {str(totalFiles)} files, {str(round((fileCounter/totalFiles *100), 1))}% ', end=' ')
    
            for i in range(4):
                yAll = dataSet[:,i]
                #loops through defined sections - sectioning is to enable easier fitting. sections are based on baseline points. 
                for j in range(len(startStopList)):
                    xStartWn = startStopList[j][0]
                    xStopWn = startStopList[j][1]
                    startIndex, stopIndex, xSection, ySection = getSection(wavenumbers, yAll, yZeros, xStartWn, xStopWn)
                    initGuess_multi, bounds_multi, popt_multi, pcov_multi, resultsArr = fitSection(xStartWn, xStopWn, xSection, ySection)
                    
                    if j!=0: #this is just to skip plotting the first setion, 565-742, in the preview window, since this section's fit is relatively straightforward. 
                        #plotFit(initGuess_multi, popt_multi, xSection, ySection, filename)
                        plotFitCheck(initGuess_multi, popt_multi, xSection, ySection, filename)
                    if j == 0:  #create the array for the fit results (for the first section)
                        allParams = resultsArr
                    else:   # add to the array for the fit results (for all following sections)
                        allParams = np.vstack([allParams, resultsArr])
                 
                columnsList = ['wavenumbers', 'area (N'+normWn+')', 'FWHM', 'y int','height (N'+normWn+')']   
                
                #dividing by all  areas and adding as new columns
                totalPeaks = allParams.shape[0]
                for ind in range(totalPeaks):
                    allParams, columnsList = divByPeak(allParams, ind, columnsList)
                    
                #adding column headers 
                outDf = pd.DataFrame(allParams, columns=columnsList)
                
                #print(f"fitting file: {filename}")
                outParamsFileName = filename[:-4].replace(' ', '') + f"-fit_{i+1}.csv"
                outputPath = chamberOutFolder / dateFolder 
                outputPath.mkdir(parents=True, exist_ok=True)
                outDf.to_csv(outputPath / outParamsFileName, index=False)
    return fileCounter

def fitMultiFiles(startHours, endHours):
    #just for progress count
    setCount = 0
    currentCount = 0 
    fileCounter = 0
    
    #just counting files for tracking
    for chFolder in normFolder.iterdir():
        for dateFolder in chFolder.iterdir():
            expHrs = int(str(dateFolder).split("\\")[-1].split("-")[-1].replace("h", ""))
            if expHrs >= startHours and expHrs <= endHours:  
                fileCounter+= len(list(dateFolder.glob("*Avg.csv")))
                #print(f' number of files in dateFolder: {len(list(dateFolder.glob("*.csv")))}')
    
    totalFileCount = fileCounter
    fileCounter = 0
    
    #actual loop
    for chFolder in normFolder.iterdir():
        for dateFolder in chFolder.iterdir():
            expHrs = int(str(dateFolder).split("\\")[-1].split("-")[-1].replace("h", ""))
            if expHrs >= startHours and expHrs <= endHours: 
                #print(expHrs)
                setCount +=1
    for chFolder in normFolder.iterdir():
        chFolderNm = str(chFolder).split("\\")[-1]
        for dateFolder in chFolder.iterdir():
            dateFolderNm = str(dateFolder).split("\\")[-1]
            expHrs = int(str(dateFolder).split("\\")[-1].split("-")[-1].replace("h", ""))
            if expHrs >= startHours and expHrs <= endHours:    
                currentCount +=1
                #print(f'\r processing set {currentCount} of {setCount}. {round(((currentCount-1)/setCount)*100, 1)}% complete', end=' ')
                try:
                    fileCounter = fitOneFile(chFolderNm, dateFolderNm, totalFileCount, fileCounter)
                except RuntimeError:
                    timeoutList.append((str(chFolder).split("\\")[-1], str(dateFolder).split("\\")[-1]))
                    print('\r' + str(timeoutList))
                    pass

folderNm='FTIR-data-PET-exposure'  
directory = "C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity"
parentDir = Path(directory)
#parentDir = Path().absolute().parent
blFilePath = parentDir / folderNm / 'PET-baseline-wns.txt'
normFTIRFolder = parentDir / folderNm / "2_normalized" 
normWn = '723'
normFolder = normFTIRFolder / normWn

fontString = "Palatino Linotype"

startStopList = [['565', '742'],['1517', '1900']]       #define regions for fitting in wns

peakDict = createPeakDict(startStopList)
print(peakDict)

#skipList = [("chamber-5", "226h")]
skipList = []
timeoutList = []

#fitOneFile("chamber-5", "20250804-1171h")

#enter start and stop hours for fitting a range of date folders
fitMultiFiles(0, 1393)
print('\n' + str(timeoutList))
