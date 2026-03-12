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
import re                   #regular expressions for list comprehension and string parsing 
import shutil               #for copying files 
import colorsys             #for setting colors automatically 

np.set_printoptions(suppress=True)


# normWn = '723' #(use max peak 1.1)
# #normWn = '1713' #(max peak 6)

# fontString = "Palatino Linotype"
# colorList = ["#000000", "#FF0000", "#FF8000", "#00FF00", "#009300", "#00C8C8", "#00A0FF", "#5E72FF", "#8000FF", "#D631FF", "#FF0000", "#FF0000", "#FF0000", "#FF8000", "#00FF00", "#009300", "#00C8C8", "#00A0FF", "#5E72FF", "#8000FF", "#D631FF", "#FF0000", "#FF0000"]
# conditionsDict = {1: "65 \N{DEGREE SIGN}C/50% RH", 5: "65 \N{DEGREE SIGN}C/0% RH"}
# peaks = [1679.8, 1713.5]
# peakListHydrox = [3430.3, 2968, 2850, 2660, 2551.8]
# peakListCarb = [1712, 1685]
# peakListRest = [1471, 1424, 1409, 1341, 1244, 1124, 1096, 1017, 969, 938, 871, 847, 723]

def getChamber(file):
    return int(re.sub('[a-zA-Z]+', '', file.split('-')[1]))

def getPosition(file):
    return int(re.sub('[a-zA-Z]+', '', file.split('-')[2]))

#make list of chamber folders 
def makeChamberFolderList(dataSourceFolder):
    chList = []
    for root, dirs, files in os.walk(dataSourceFolder):
          for file in files:
            chamberFolder = Path(root).parent
            if chamberFolder not in chList:
                chList.append(chamberFolder)
    return chList

#creates list of all positions for which there is data in the folder. 
# accepts chamber folder as full path
#returns list of positions in folder as ints 
#Limitations and/or to-dos: 
#dependent on file name format as written 
#still using date folder variable name conventions and hard coding file levels 
def createPosList(chFolder):                                
    posList = []
    for dateFolder in chFolder.iterdir():
        for filename in os.listdir(dateFolder):
            if 'Avg' not in filename and 'Back' not in filename:  
                posNum = int(filename.split("-")[2].replace("Pos", "").replace("pos", ""))
                if posNum not in posList: 
                    posList.append(posNum)
    posList.sort()
    return posList

# #loads data file. file is csv with 1 wavenumber column (x) and 4 spectra columns (y), one for each measuremennt 1-4 in order 
def getSpec(filepath):
    file = np.loadtxt(filepath, delimiter=',')   #gets file with wn column and all 4 spectra columns
    # wns = file[:,0]                                         #pulls only wn column
    # allSpec = np.delete(file, 0, 1)                         #deletes wn column leaving only 4 spectra columns
    return file

# #iterates through all existing measurments dates to find the measurement (1-4) that most frequently best 
# #the average of the 4 measurements. Ultiamtely, the same measurement number will be used for all plots for one posiition. 
def findRepSpecIndex(filelist, path):
        repSpecIndList = []
        fileList = []
        
        for file in filelist: 
                normSet = getSpec(path / file)
                yData = np.delete(np.copy(normSet), 0, 1)
                rows, columns = yData.shape
                normAvg = yData.mean(axis=1)
                meanDiffList = [(normAvg[1] -yData[:,i]).mean(axis=0) for i in range(columns)]        #for each data set, find replicate closest to the average 
                repSpecIndList.append(meanDiffList.index(min(meanDiffList)))
        repSpecInd = max(set(repSpecIndList), key=repSpecIndList.count)             #find which index is closest to the average the most often
        return repSpecInd

# def createNames(chFolder, fileList, pos, chDatOutFolder):
#     #naming plot
#     chamberID = str(chFolder)[-1]                           #for 6-port specifically - if chamber label is more digits, will need to change. 
#     condStr = conditionsDict[int(chamberID)]
#     targetT = str(int(filtersPctDF.loc[pos]["Target pct T"]))
#     actualT = str(round(filtersPctDF.loc[pos]["Chamber " + str(chamberID) + " UV Vis pct T"], 2))
#     plotTitle = f"{condStr}\n{targetT}% T (Actual: {actualT}% T)" 
#     #naming file
#     mat = (fileList[-1].split("-")[0].split("_")[1])
#     ID = ("-".join(fileList[-1].split("-")[1:3]))
#     airBack = (fileList[-1].split("-")[-1].split("_")[0])
#     outDataFilename = ("-".join([mat, ID, airBack])) + "-N" + str(normWn) + ".csv"
#     outDataFilePath = chDatOutFolder / outDataFilename
#     return plotTitle, outDataFilename, outDataFilePath   

def makeSpecArr(filelist, path, repSpecInd): 
        wns = getSpec(path / filelist[0])[:, 0]
        dataSet = np.column_stack((wns, np.array([getSpec(path / file)[:,repSpecInd+1] for file in filelist]).T ))#array of data column for representative spectra replicate index at all time points, with wavenumbers at front         
        return dataSet


# def plotSpec(specDF, plotTitle, axisLims, dims, outFileNm, outFolder, pkInds, stepSize, peakList):   
#     if len(specDF.columns)>2:
#         pltFileNm = f"{outFileNm[:-4]}_{axisLims[0]}-{axisLims[1]}.png"
#         #outPath = outFolder / f"{axisLims[0]}-{axisLims[1]}" / pltFileNm
#         outPath = outFolder / pltFileNm
#         #outPath.mkdir(parents=True, exist_ok=True)
#         figSpec, specX = plt.subplots(figsize=dims, dpi=300, layout='constrained')
#         plt.title(plotTitle, fontsize=22, y=1.05, family = fontString)
#         specX.axis(axisLims)
#         specX.xaxis.set_inverted(True)
#         specX.tick_params(axis='both', which='major', labelsize=22)
#         #plt.xlabel("wavenumbers (cm\u207b\u2071)")
#         specX.set_xlabel("Wavenumbers (cm\u207b\u00B9)", fontsize=22, font = fontString)
#         specX.set_ylabel(f"Absorbance (norm. to {normWn} cm\u207b\u00B9)", fontsize=21, font = fontString)
#         #plt.ylabel(f"Absorbance (normalized to {normWn})")
#         plt.xticks(np.arange(axisLims[0], axisLims[1]+1, stepSize))
#         for tick in specX.get_xticklabels():
#             tick.set_fontname(fontString)
#         for tick in specX.get_yticklabels():
#             tick.set_fontname(fontString)
#         numSpec = len(specDF.columns)
#         #interval = round(numSpec-1/2, 0)
#         for i in range(1, numSpec):
#             if numSpec<4 or i % 3 == 0 or i==1 or i==numSpec-1:
#                 expHr = (specDF.columns[i]).split("-")[-1]
#                 dose = doseCSV[str(position)][expHr]
#                 wns = specDF[specDF.columns[0]]
#                 spectrum = specDF[specDF.columns[i]]
#                 specX.plot(wns, spectrum, c=colorList[i-1], linewidth=1.5, label=f"{round(dose, 1)} W/m\u00B2") 
#         #specX.plot(wns[pkInds], spectrum[pkInds],'|', ms = 10)
#         for i in range(len(peakList)):
#             labelIndex = wns.tolist().index(min(wns, key=lambda x:abs(x-peakList[i])))
#             #specX.annotate(peakList[i], xy=(peakList[i], spectrum[labelIndex]+0.07), xytext=(peakList[i],spectrum[labelIndex]+0.2), arrowprops=dict(width=0.5, headwidth=4, headlength=3, color = 'black'), ha='center', rotation="vertical", fontsize=16, family = fontString)
#             #specX.annotate(peakList[i], xy=(peakList[i], spectrum[labelIndex]+0.02), xytext=(peakList[i],spectrum[labelIndex]+0.05), arrowprops=dict(width=0.5, headwidth=4, headlength=3, color = 'black'), ha='center', rotation="vertical", fontsize=16, family = fontString)
#             specX.annotate(peakList[i], xy=(peakList[i], spectrum[labelIndex]+0.12), xytext=(peakList[i],spectrum[labelIndex]+0.3), arrowprops=dict(width=0.5, headwidth=4, headlength=3, color = 'black'), ha='center', rotation="vertical", fontsize=16, family = fontString)
#         specX.legend(loc='upper right', bbox_to_anchor=(1.01, 0.999), frameon=False, prop=dict(family=fontString, size=16))
#         figSpec.savefig(outPath, format='png', dpi=300, transparent=True)
        
#         #plt.close(figSpecFull)
        
# filtersPctDF = pd.read_csv(parentDir / "filters-pct-T.csv", sep=',', header=0, index_col=0, on_bad_lines="skip", engine='python')

#returns list of evenly-spaced colors to use for plot 
def genPlotColors(dataSet):
    totColors = len(dataSet)-1
    hlsTups = [(x/totColors, 0.4, 1) for x in range(totColors)]
    rgbTups = list(map(lambda x: colorsys.hls_to_rgb(*x), hlsTups))
    rgbTups.insert(0, '0')
    return rgbTups

def formatData(
        file,
        parDir,
        dataCols,
        index
    ):
    print(index)
    chID = getChamber(file)
    position = getPosition(file)
    expHr = (re.findall("\d{1,4}[Hh]", file))[0]      # https://stackoverflow.com/questions/45629069/check-if-a-string-contains-any-amount-of-numbers-followed-by-a-specific-letter
    doseCSV = pd.read_csv(Path(parDir).parent / f"Doses-Ch-{chID}.csv", sep=',', header=0, index_col="file suffix" , on_bad_lines="skip", engine='python')
    dose = doseCSV[str(position)][expHr]

    colorList = genPlotColors(dataCols)
    pltCol = colorList[index-1]
    
    legTxt=f"{round(dose, 1)} W/m\u00B2"

    



    # if filHandle == True: 
    #     pltCol, fillSty, legTxt = filtersFormat(position, allPos, chID)

    # return pltCol, fillSty, legTxt
    return pltCol, legTxt

def formatPlot(
        posPlot,
        filename,  
        fontString = "Palatino Linotype"
    ):
    
    
    # #title for plot - in-progress  
    chID = getChamber(filename)
    position = getPosition(filename)
    titleString = f"{filename}"
    # "cm\u207b\u00B9"
    
    fmtPlot = posPlot
    fmtPlot.legend(loc='upper left', bbox_to_anchor=(0.99, 0.99), frameon=False, prop=dict(family=fontString, size=18))
    plt.title(titleString, fontsize=22, y=1.05, family = fontString)
    # # trendsX.set_xlabel(f"Dose (W/m\u00B2)", fontsize=22, font = fontString)
    # # #trendsX.set_ylabel(f"Area ({peakDict[targetPeak]}) / Area ({normWn})", fontsize=22, font = fontString)
    # # trendsX.set_ylabel(f"\u0394 Product index ({peakDict[targetPeak]} cm\u207b\u00B9)", fontsize=22, font = fontString)
    # # trendsX.tick_params(axis='both', which='major', labelsize=22)
    return fmtPlot

#arranges normalized data in folders by position for easier iterating

def normByPosition(
    rawDataDir,
    normFTIRFolder = "2_normalized",
    byPosFolder = "2a_norm-by-position",
    normWn = '723',
    ):

    rawDataFolder = Path(rawDataDir)
    parentDir = rawDataFolder.absolute().parent 
    sourceFolder = parentDir / normFTIRFolder / normWn
    byPosOutFolder = parentDir / byPosFolder / normWn

    chFolderList = makeChamberFolderList(sourceFolder)

    for chamberFolder in chFolderList:
        positionList = createPosList(chamberFolder)
        chamberID = str(chamberFolder).split("\\")[-1].split('-')[-1]

        for root, dirs, files in os.walk(chamberFolder):
            for filename in files:
                if 'Avg' not in filename and 'Back' not in filename:               #currently not plotting averages or back spectra
                        filePos = filename.split('-')[2]
                        outPosFolder = byPosOutFolder / f"chamber-{chamberID}" / filePos 
                        outPosFolder.mkdir(parents=True, exist_ok=True)
                        shutil.copy(Path(root) / filename, outPosFolder / filename)


def createPlotData(
        rawDataDir,
        normWn = '723',
        filtersFileNm = "filters-pct-T.csv",
        normByPosFolderNm = "2a_norm-by-position",
        repSpecOutFolderNm = "2b_repSpecByPosition"
        ):
    
        #find folder with normalized data 
        rawDataFolder = Path(rawDataDir)
        parentDir = rawDataFolder.absolute().parent 
        normByPosFolder = parentDir / normByPosFolderNm / normWn
        repSpecOutFolder = parentDir / repSpecOutFolderNm / normWn
        filterListPath = parentDir / filtersFileNm

        for root, dirs, files in os.walk(normByPosFolder):  
                if len(files)>0:
                        chamberFolder = Path(root).parent
                        outputFolder = Path(str(chamberFolder).replace(str(normByPosFolder), str(repSpecOutFolder)))    #replace source folder path with target folder path in chamber folder (do not need or want position folders to be preserved)
                        outputFolder.mkdir(parents=True, exist_ok=True)  
                        expHr = (re.findall("\d{1,4}[Hh]", files[0]))[0]      # https://stackoverflow.com/questions/45629069/check-if-a-string-contains-any-amount-of-numbers-followed-by-a-specific-letter
                        newFileNm = files[0].replace(f"{expHr}", "plot")

                        # chamberID = getChamber(files[0])
                        # doseCSV = pd.read_csv(parentDir.parent / f"Doses-Ch-{chamberID}.csv", sep=',', header=0, index_col="file suffix" , on_bad_lines="skip", engine='python')
                        # posFolder = Path(root)
                        # position = getPosition(files[0])

                        
                        repSpecIndex = findRepSpecIndex(files, Path(root))
                          
                        specArr = makeSpecArr(files, Path(root), repSpecIndex) #one position and one chamber only!! change from previous iteration 
                        colNamesList = [f"{file} ({repSpecIndex+1})" for file in files]
                        colNamesList.insert(0, 'wavenumbers')
                        columnNames = ",".join(colNamesList)
                        np.savetxt(Path(outputFolder / newFileNm), specArr, '%5.7f', delimiter=',', header=columnNames)



##plotting formatting in progress! re-working old, clunky script into more flexible and clean, user-friendly script.  large chunks
##of commented functions just kept to refer to for plotting and formatting  
def plotSpectra(
        rawDataDir,
        normWn = '723',
        filtersFileNm = "filters-pct-T.csv",
        repSpecFolderNm = "2b_repSpecByPosition"
        ):
     
        #find folder with plot-ready data
        rawDataFolder = Path(rawDataDir)
        parentDir = rawDataFolder.absolute().parent 
        repSpecFolder = parentDir / repSpecFolderNm / normWn
        filterListPath = parentDir / filtersFileNm

        for root, dirs, files in os.walk(repSpecFolder):  
            if len(files)>0:
                for file in files: 
                    columnNames = pd.read_csv(Path(root) / file, delimiter=',').columns.tolist()
                    plotData = getSpec(Path(root) / file)
                       

                    figure, positionPlot = plt.subplots(figsize=(6, 6), dpi=300)                 #create plot
                    wns = plotData[:,0]

                    #reducing number of columns to plot to prevent crowding 
                    numSpecTotal = len(columnNames)-1          
                    plotColumns = [i for i in range(1, len(columnNames)) if numSpecTotal<4 or i % 3 == 0 or i==1 or i==numSpecTotal-1]           

                    for col in plotColumns: 
                        

                        filename, replicate = columnNames[col].split(' ')[0],  columnNames[col].split(' ')[1].replace('(', '').replace(')', '')

                        yData = plotData[:, col]

                        #set up parameters for formatting DATA (data as in lines, markers)
                        # lineColor, fillStyle, legendText = formatData(filename, root)
                        lineColor, legendText = formatData(filename, parentDir, columnNames, col)
                        positionPlot.plot(wns, yData,
                            linestyle='-', label=legendText,
                            color=lineColor) 
                                # , markeredgecolor=lineColor, markerfacecolor=lineColor,
                                # marker='o', fillstyle=fillStyle, markersize= 12, 
                                # ) 
                    
                    #formatting PLOT: figure, title, shape, axes, etc 
                    positionPlot = formatPlot(positionPlot, file)
                    plt.show()
            

                            

if __name__ == "__main__":          #does not run if importing only if running
    # normByPosition("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr/0_raw-data")
    # createPlotData("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr/0_raw-data")
    plotSpectra("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-ND-filters-ATR-corr/0_raw-data")