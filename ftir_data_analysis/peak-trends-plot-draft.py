
# -*- coding: utf-8 -*-
"""

@author: klj
"""

import os, itertools  #os needed for os.walk. itertools allows for listcomprehension, speficially itertools.groupby allows easy grouping of replicate files 
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re #regular expressions for list comprehension and string parsing 
#import math

#when plotting by filters, sorts files for iterating into order grouped by filter level to make plotting easier/nicer. 
def filterSort(
        chFiles 
    ): 
    targetTs = np.unique(filtersArray[:,1])
    sortFileList = []
    for targetT in targetTs:
        sameTargetT = filtersArray[np.where(filtersArray[:,1]==targetT)]
        for file in chFiles:
           position = float(re.sub('[a-zA-Z]+', '', file.split('-')[2]))
           if position in sameTargetT: 
               sortFileList.append(file)
               
    return sortFileList

def filtersFormat(
        pos, 
        allPos, 
        chID #for finding actual %T, not included yet 
    ):
        targetTs = np.unique(filtersArray[:,1]).tolist()
        targetT = filtersArray[np.where(filtersArray[:,0]==pos)][:,1][0]
        sameTargetTs = [posID for posID in allPos if posID in filtersArray[np.where(filtersArray[:,1]==targetT)][:,0].tolist()]
        colorList = ["#000000", "#D631FF", "#4FBDFF", "#F4A666", "#349F9F"]
        markerList = ["o", "s", "D"]
        fillList = ["full", "left", "none"]
        filterColor = colorList[targetTs.index(targetT)]
        replicateFill = fillList[sameTargetTs.index(pos)]
        
        #needs update to include actual %T not target
        TEMPLEGEND = f"~{targetT}% T"

        return filterColor, replicateFill, TEMPLEGEND

def formatData(
        avgFile, 
        avgFolder,
        allPos, 
        filHandle
    ):
    chID = str(Path(avgFolder).parent.parent).split('\\')[-1].split('-')[-1]
    position = float(re.sub('[a-zA-Z]+', '', avgFile.split('-')[2]))
    if filHandle == True: 
        pltCol, fillSty, legTxt = filtersFormat(position, allPos, chID)

    return pltCol, fillSty, legTxt

#copied from another script, needs editing to work in this one
# def genPlotColors():
#     totPks = len(pkDict)
#     hlsTups = [(x/totPks, 0.4, 1) for x in range(totPks)]
#     rgbTups = list(map(lambda x: colorsys.hls_to_rgb(*x), hlsTups))
#     return rgbTups

def formatPlot(
        chPlot, 
        avgFolder, 
        propFolderNm, 
        pkWn,
        fontString = "Palatino Linotype"
    ):
    
    #title for plot - in-progress  
    chID = str(Path(avgFolder).parent.parent).split('\\')[-1].split('-')[-1]
    titleString = f"Chamber {chID}: {propFolderNm} {pkWn} cm\u207b\u00B9" 
    
    fmtPlot = chPlot
    plt.title(titleString, fontsize=22, y=1.05, family = fontString)
    # trendsX.legend(loc='upper left', bbox_to_anchor=(0.99, 0.99), frameon=False, prop=dict(family=fontString, size=18))
    # trendsX.set_xlabel(f"Dose (W/m\u00B2)", fontsize=22, font = fontString)
    # #trendsX.set_ylabel(f"Area ({peakDict[targetPeak]}) / Area ({normWn})", fontsize=22, font = fontString)
    # trendsX.set_ylabel(f"\u0394 Product index ({peakDict[targetPeak]} cm\u207b\u00B9)", fontsize=22, font = fontString)
    # trendsX.tick_params(axis='both', which='major', labelsize=22)
    return fmtPlot


def plotChambers(
    rawDataDir,
    plotFolderNm,
    inputPkWn,
    normWn = '723',
    propertyFolder = 'area_N723', #TO-DO: remove hard coding 
    filtersFile = None,
    filtersHandling = True
    ):
    #setting up useful paths
    rawDataFolder = Path(rawDataDir)
    parentDir = rawDataFolder.absolute().parent 
    plotFolder = parentDir / plotFolderNm / normWn
    

    #getting filters array 
    if filtersHandling == True:
        global filtersArray, filtersColNames
        filtersArray = np.loadtxt(parentDir.parent / filtersFile, delimiter=',', skiprows=1)
        filtersColNames = np.genfromtxt(parentDir.parent / filtersFile, delimiter=',', names=True, max_rows=1).dtype.names

    for root, dirs, files in os.walk(plotFolder):
        #should do this sequence below once for each chamber. 

        if len(files)!=0 and root.split('\\')[-1] == 'Average' and root.split('\\')[-2] == propertyFolder: #only going through the terminal level folders named averages

            wavenumbers = (np.genfromtxt(Path(root) / files[0], delimiter=',', names=True, max_rows=1).dtype.names) #retrieving column names. genfromtxt stores column names as dtype. skips column header for expHrs
            peakWn = min(wavenumbers[2:], key=lambda x: abs(int(x)-inputPkWn))        #find closest wavenumber in data to input peak wavenumber 
            yColInd = wavenumbers.index(peakWn)                                       #uses input peak number to find data column

            figure, chamberPlot = plt.subplots(figsize=(6, 6), dpi=300)                 #create chamber plot
            
            #need to move filters Array somewhere it can be turned on and off 
            sortedFiles = filterSort(files) if filtersHandling==True else files                          #sort files by filter grouping before iterating to simplify
            allPositions = [float(re.sub('[a-zA-Z]+', '', file.split('-')[2])) for file in files]

            for file in sortedFiles:
                #removes position 1 when plotting by dose. should come up with a way to easily include this when plotting by exp time
                if float(re.sub('[a-zA-Z]+', '', file.split('-')[2])) != 1:

                    avgPath = Path(root) / file
                    stDevPath = Path(str(root).replace('Average', 'StDev')) / file.replace('Avg', 'StDev')

                    avgsArray = np.loadtxt(avgPath, delimiter=',', skiprows=1)
                    stDevArray = np.loadtxt(stDevPath, delimiter=',', skiprows=1)
                    
                    xColData = avgsArray[:, 1]
                    yColData = avgsArray[:,yColInd]

                    # formatData(file, root, propertyFolder, peakWn, filtersArray, filtersColNames)
                    plotColor, fillStyle, legendText = formatData(file, root, allPositions, filtersHandling)
                    chamberPlot.plot(xColData, yColData,
                                    linestyle='-', 
                                    color=plotColor, markeredgecolor=plotColor, markerfacecolor=plotColor,
                                    marker='o', fillstyle=fillStyle, markersize= 12, 
                                    label=legendText)
            
            chamberPlot = formatPlot(chamberPlot, root, propertyFolder, peakWn)
            plt.show()
            
                    

if __name__ == "__main__":          #does not run if importing only if running
    plotChambers("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-exposure/0_raw-data", plotFolderNm = "5a_fit-avg-std_by-prop_no-corr", inputPkWn = 1711, filtersFile = 'filters-pct-T.csv')