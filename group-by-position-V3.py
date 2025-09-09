# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:21:30 2025

@author: klj
"""

#groups representative spectra by position and plots spectra over time. still needs to be set up to 
#output these as files. 

import os 
from pathlib import Path
import numpy as np
import pandas as pd
#from scipy.signal import find_peaks as find_peaks
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)
folderNm='FTIR-data-PET-exposure'  
parentDir = Path().absolute().parent
normWn = '723' #(use max peak 1.1)
#normWn = '1508'
#normWn = '1713' #(max peak 6)
normFTIRFolder = parentDir / folderNm / "2_normalized" / normWn
filterListPath = parentDir / "filters-pct-T.csv"

fontString = "Palatino Linotype"
colorList = ["#000000", "#FF0000", "#FF8000", "#00FF00", "#009300", "#00C8C8", "#00A0FF", "#5E72FF", "#8000FF", "#D631FF", "#FF0000", "#FF0000", "#FF0000", "#FF8000", "#00FF00", "#009300", "#00C8C8", "#00A0FF", "#5E72FF", "#8000FF", "#D631FF", "#FF0000", "#FF0000"]
conditionsDict = {1: "65 \N{DEGREE SIGN}C/50% RH", 5: "65 \N{DEGREE SIGN}C/0% RH"}
peaks = [1679.8, 1713.5]
peakListHydrox = [3430.3, 2968, 2850, 2660, 2551.8]
peakListCarb = [1712, 1685]
peakListRest = [1471, 1424, 1409, 1341, 1244, 1124, 1096, 1017, 969, 938, 871, 847, 723]

#creates list of all positions currently included in measurements. 
def createPosList(chFolder):
    posList = []
    for dateFolder in chFolder.iterdir():
        for filename in os.listdir(dateFolder):
            #print(filename)
            if filename[-7:-4] != "Avg" and filename.split("-")[4].split("_")[0]=="Air":
                posNum = int(filename.split("-")[2].replace("Pos", "").replace("pos", ""))
                if posNum not in posList: 
                    posList.append(posNum)
    posList.sort()
    return posList

#loads data file. file is csv with 1 wavenumber column (x) and 4 spectra columns (y), one for each measuremennt 1-4 in order 
def getSpec(fileNm, dateFolder):
    file = np.loadtxt(dateFolder / fileNm, delimiter=',')   #gets file with wn column and all 4 spectra columns
    wns = file[:,0]                                         #pulls only wn column
    allSpec = np.delete(file, 0, 1)                         #deletes wn column leaving only 4 spectra columns
    return wns, allSpec

#iterates through all existing measurments dates to find the measurement (1-4) that most frequently best 
#the average of the 4 measurements. Ultiamtely, the same measurement number will be used for all plots for one posiition. 
def findRepSpec(pos, chFolder):
    repSpecIndList = []
    fileList = []
    for dateFolder in chFolder.iterdir():
        for filename in os.listdir(dateFolder):
            filePos = filename.split("-")[2].replace("Pos", "").replace("pos", "")  #collects position number from file name 
            airBack = filename.split("-")[4].split("_")[0]                          #collects air v back designation from file name
            if filePos == str(pos) and filename[-7:-4] != "Avg" and airBack=="Air": #currently this script ONLY PLOTS AIR. this will need to be addressed if back spectra are desired. 
                fileList.append(filename)                                           #if the file matches the position being collected, the filename is added to the list for future file access. 
                wns, normSet = getSpec(filename, dateFolder)
                rows, columns = normSet.shape
                normAvg = normSet.mean(axis=1)
                meanDiffList = []
                for i in range(columns):
                    diff = normAvg[1] - normSet[:,i]                            #for each data set, find replicate closest to the average 
                    meanDiffList.append(diff.mean(axis=0))
                repSpecIndList.append(meanDiffList.index(min(meanDiffList)))
    repSpecInd = max(set(repSpecIndList), key=repSpecIndList.count)             #find which index is closest to the average the most 
    
    return repSpecInd, fileList, wns

def createNames(chFolder, fileList, pos, chDatOutFolder):
    #naming plot
    chamberID = str(chFolder)[-1]                           #for 6-port specifically - if chamber label is more digits, will need to change. 
    condStr = conditionsDict[int(chamberID)]
    targetT = str(int(filtersPctDF.loc[pos]["Target pct T"]))
    actualT = str(round(filtersPctDF.loc[pos]["Chamber " + str(chamberID) + " UV Vis pct T"], 2))
    plotTitle = f"{condStr}\n{targetT}% T (Actual: {actualT}% T)" 
    #naming file
    mat = (fileList[-1].split("-")[0].split("_")[1])
    ID = ("-".join(fileList[-1].split("-")[1:3]))
    airBack = (fileList[-1].split("-")[-1].split("_")[0])
    outDataFilename = ("-".join([mat, ID, airBack])) + "-N" + str(normWn) + ".csv"
    outDataFilePath = chDatOutFolder / outDataFilename
    return plotTitle, outDataFilename, outDataFilePath   

def createPlotDF(fileList, repSpecInd, wns, chFolder):
    specDF = pd.DataFrame(wns, columns=['wavenumbers'])    
    for dateFolder in chFolder.iterdir():
        for filename in os.listdir(dateFolder):
            if filename in fileList:
                wavenumbers, normSet = getSpec(filename, dateFolder)
                columnHeader = "-".join(filename.split("-")[1:4])
                specDF[columnHeader] = normSet[:,repSpecInd]            
    return specDF
   
def findPeaks(pkWns, wns):
    pkIndsList = []
    for i in pkWns:
        pkIndTemp = wns.tolist().index(min(wns.tolist(), key=lambda x:abs(i-x)))
        pkIndsList.append(pkIndTemp)

    return pkIndsList

def plotSpec(specDF, plotTitle, axisLims, dims, outFileNm, outFolder, pkInds, stepSize, peakList):   
    if len(specDF.columns)>2:
        pltFileNm = f"{outFileNm[:-4]}_{axisLims[0]}-{axisLims[1]}.png"
        #outPath = outFolder / f"{axisLims[0]}-{axisLims[1]}" / pltFileNm
        outPath = outFolder / pltFileNm
        #outPath.mkdir(parents=True, exist_ok=True)
        figSpec, specX = plt.subplots(figsize=dims, dpi=300, layout='constrained')
        plt.title(plotTitle, fontsize=22, y=1.05, family = fontString)
        specX.axis(axisLims)
        specX.xaxis.set_inverted(True)
        specX.tick_params(axis='both', which='major', labelsize=22)
        #plt.xlabel("wavenumbers (cm\u207b\u2071)")
        specX.set_xlabel("Wavenumbers (cm\u207b\u00B9)", fontsize=22, font = fontString)
        specX.set_ylabel(f"Absorbance (norm. to {normWn} cm\u207b\u00B9)", fontsize=21, font = fontString)
        #plt.ylabel(f"Absorbance (normalized to {normWn})")
        plt.xticks(np.arange(axisLims[0], axisLims[1]+1, stepSize))
        for tick in specX.get_xticklabels():
            tick.set_fontname(fontString)
        for tick in specX.get_yticklabels():
            tick.set_fontname(fontString)
        numSpec = len(specDF.columns)
        #interval = round(numSpec-1/2, 0)
        for i in range(1, numSpec):
            if numSpec<4 or i % 3 == 0 or i==1 or i==numSpec-1:
                expHr = (specDF.columns[i]).split("-")[-1]
                dose = doseCSV[str(position)][expHr]
                wns = specDF[specDF.columns[0]]
                spectrum = specDF[specDF.columns[i]]
                specX.plot(wns, spectrum, c=colorList[i-1], linewidth=1.5, label=f"{round(dose, 1)} W/m\u00B2") 
        #specX.plot(wns[pkInds], spectrum[pkInds],'|', ms = 10)
        for i in range(len(peakList)):
            labelIndex = wns.tolist().index(min(wns, key=lambda x:abs(x-peakList[i])))
            #specX.annotate(peakList[i], xy=(peakList[i], spectrum[labelIndex]+0.07), xytext=(peakList[i],spectrum[labelIndex]+0.2), arrowprops=dict(width=0.5, headwidth=4, headlength=3, color = 'black'), ha='center', rotation="vertical", fontsize=16, family = fontString)
            #specX.annotate(peakList[i], xy=(peakList[i], spectrum[labelIndex]+0.02), xytext=(peakList[i],spectrum[labelIndex]+0.05), arrowprops=dict(width=0.5, headwidth=4, headlength=3, color = 'black'), ha='center', rotation="vertical", fontsize=16, family = fontString)
            specX.annotate(peakList[i], xy=(peakList[i], spectrum[labelIndex]+0.12), xytext=(peakList[i],spectrum[labelIndex]+0.3), arrowprops=dict(width=0.5, headwidth=4, headlength=3, color = 'black'), ha='center', rotation="vertical", fontsize=16, family = fontString)
        specX.legend(loc='upper right', bbox_to_anchor=(1.01, 0.999), frameon=False, prop=dict(family=fontString, size=16))
        figSpec.savefig(outPath, format='png', dpi=300, transparent=True)
        
        #plt.close(figSpecFull)
        
filtersPctDF = pd.read_csv(parentDir / "filters-pct-T.csv", sep=',', header=0, index_col=0, on_bad_lines="skip", engine='python')


for chamberFolder in normFTIRFolder.iterdir():
    chamberID = str(chamberFolder).split("\\")[-1].split("-")[-1]
    doseCSV = pd.read_csv(parentDir / f"Doses-Ch-{chamberID}.csv", sep=',', header=0, index_col="file suffix" , on_bad_lines="skip", engine='python')
    print(f"chamberID: {chamberID}")
    chamberDataOutFolder = parentDir / folderNm / "x_for-plot" / str(normWn) /str(str(chamberFolder).split('\\')[-1:][0])
    chamberDataOutFolder.mkdir(parents=True, exist_ok=True)
    chamberFigOutFolder = parentDir / folderNm / "x_figs" / str(normWn) /str(str(chamberFolder).split('\\')[-1:][0])
    chamberFigOutFolder.mkdir(parents=True, exist_ok=True)
    positionList = createPosList(chamberFolder)
    #dosesDF = pd.read_csv(parentDir / f"Doses-Ch-{str(chamberFolder)[-1]}.csv", sep=',', header=0, index_col=0,  usecols=["file suffix", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17"], on_bad_lines="skip", engine='python')
    for position in positionList:  
        
        repSpecIndex, posFileList, wavenumbers = findRepSpec(position, chamberFolder)
        pltTitle, outDatFileNm, outDataPath = createNames(chamberFolder, posFileList, position, chamberDataOutFolder)
        peakInds = findPeaks(peaks, wavenumbers)
        spectraPlotDF = createPlotDF(posFileList, repSpecIndex, wavenumbers, chamberFolder)

        spectraPlotDF.to_csv(outDataPath, sep=',')
        #plotSpec(spectraPlotDF, [400, 4000, -0.05, 6])
        #723
        #plotSpec(spectraPlotDF, pltTitle, [1440, 1850, -0.05, 1.5], (5,6), outDatFileNm, chamberFigOutFolder, peakInds, 100, peakListCarb)
        #plotSpec(spectraPlotDF, pltTitle, [2250, 4000, -0.01, 0.42], (5,6), outDatFileNm, chamberFigOutFolder, peakInds, 500, peakListHydrox)
        plotSpec(spectraPlotDF, pltTitle, [400, 1500, -0.05, 1.55], (10,6), outDatFileNm, chamberFigOutFolder, peakInds, 100, peakListRest)
        
        
        #plotSpec(spectraPlotDF, pltTitle, [2300, 4000, -0.01, 0.2], (10,5), outDatFileNm, chamberFigOutFolder, peakInds)
        #plotSpec(spectraPlotDF, pltTitle, [400, 1850, -0.05, 1.1], (10,5), outDatFileNm, chamberFigOutFolder, peakInds)
        #1409
        #plotSpec(spectraPlotDF, pltTitle, [1550, 1850, -0.05, 5.5], (5,5), outDatFileNm, chamberFigOutFolder, peakInds)
        #plotSpec(spectraPlotDF, pltTitle, [400, 4000, -0.05, 6], (10,5), outDatFileNm, chamberFigOutFolder, peakInds)
        #plotSpec(spectraPlotDF, pltTitle, [2300, 4000, -0.01, 0.2], (10,5), outDatFileNm, chamberFigOutFolder, peakInds)
        #plotSpec(spectraPlotDF, pltTitle, [400, 1850, -0.05, 6], (10,5), outDatFileNm, chamberFigOutFolder, peakInds)

