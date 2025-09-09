# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 16:40:19 2025

@author: klj
"""

import os 
from pathlib import Path
import numpy as np
import pandas as pd
#from scipy.signal import find_peaks as find_peaks
#from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
#import math

np.set_printoptions(suppress=True)
folderNm='FTIR-data-PET-exposure'  
parentDir = Path().absolute().parent
normWn = '723'
trendsFTIRFolder = parentDir / folderNm / "4_peak-trends" / normWn

filterListPath = parentDir / "filters-pct-T.csv"
filtersPctDF = pd.read_csv(parentDir / "filters-pct-T.csv", sep=',', header=0, index_col=0, on_bad_lines="skip", engine='python')

trendType = "Diff"
targetProperty = "divided-Area-by1714"
targetPeak = 8

if "divided-Area" in targetProperty:
    ratioVAbs = "Ratio"
    divisor = targetProperty.split("-")[-1].replace("by", "")
else: 
    divisor = ""
    ratioVAbs = "Abs"
#targetProperty = "wavenumbers"
#numMeas = 4


conditionsDict = {1: [65, 50], 5: [65, 0]}
peakDict = {0: "631", 1: "652", 2: "689", 3: "711", 4: "725", 5: "1578", 6: "1610", 7: "1650", 8: "1685", 9: "1714", 10: "1727", 11: "1735", 12: "1749", 13: "1777", 14: "1857"}

#colorList = ["#000000", "#FF0000", "#FF8000", "#00FF00", "#009300", "#00C8C8"]
colorList = ["#000000", "#D631FF", "#4FBDFF", "#F4A666", "#349F9F"]
markerList = ["o", "s", "D"]
fillList = ["full", "left", "none"]
fontString = "Palatino Linotype"
#fontString = "Corbel"

def targetPropFunction(fullDF):
    #print(fullDF)
    newDF = fullDF["9"].div(fullDF["6"], axis=0).to_frame()
    return newDF

#creates list of all positions for which there is data in the folder. 
def createPosList():
    posList = []
    for chFolder in trendsFTIRFolder.iterdir():
        for filename in os.listdir(chFolder / targetProperty / trendType):
            posNum = int(filename.split("-")[3].replace("Pos", "").replace("pos", ""))
            #print(posNum)
            if posNum not in posList: 
                posList.append(posNum)
    posList.sort()
    return posList

def filtersList():
    targetTListAll = filtersPctDF["Target pct T"].tolist()
    targetTList = []
    for pctT in targetTListAll:
        if pctT not in targetTList:
            targetTList.append(pctT)
    targetTList.sort()
    return targetTList 

def createFilterGroup(pctT, chFolder):
    pctPosList = []
    for position in positionList:
        #print(f"{pctT}% T, {position}, {filtersPctDF.loc[position]['Target pct T']}")
        if pctT == int(filtersPctDF.loc[position]["Target pct T"]):
            #print(f"{pctT}% T, {position}")
            pctPosList.append(position)
    pctPosList.sort()
    fileList = []
    for position in pctPosList:
        fileSetNm = findPositionFiles(position, chFolder)
        fileList.append(fileSetNm)
    return fileList
            
#finds file name (except meas number and extension) containing data for each position.
def findPositionFiles(pos, chFolder):
    for filename in os.listdir(chFolder / targetProperty / trendType):
        filePos = filename.split("-")[3].replace("Pos", "").replace("pos", "")  #collects position number from file name 
        airBack = filename.split("-")[4]
        if filePos == str(pos) and airBack == "Air":
            fileSet = filename[:-8]
    #actT = str(round(filtersPctDF.loc[pos]["Chamber " + str(chamberID) + " UV Vis pct T"], 2))
    return fileSet

def getData(chFolder, fileSetNm, rep, tgtPctT):
    avgCSV = pd.read_csv(chFolder / targetProperty / trendType / f"{fileSetNm}-Avg.csv", sep=',', header=0, on_bad_lines="skip", engine='python', index_col=0)
    stdCSV = pd.read_csv(chFolder / targetProperty / trendType / f"{fileSetNm}-Std.csv", sep=',', header=0, on_bad_lines="skip", engine='python', index_col=0)
    #print(avgCSV)
    #tempAvgCol = avgCSV[str(targetPeak)].to_frame()
    #empStdCol = stdCSV[str(targetPeak)].to_frame()
    tempDF = pd.concat([avgCSV[str(targetPeak)], stdCSV[str(targetPeak)]], axis=1, )
    
    #print(tempDF)
    tempDF.columns = ["Avg", "Std"]
    pos = int(fileSetNm.split("-")[3].replace("Pos", "").replace("pos", ""))
    #print(pos)
    tempDF["pctT"] = str(round(filtersPctDF.loc[pos]["Chamber " + str(chamberID) + " UV Vis pct T"], 2))
    tempDF["targetT"] = tgtPctT
    tempDF["replicate"] = rep
    tempDF["temperature"] = conditionsDict[int(chamberID)][0]
    tempDF["RH"] = conditionsDict[int(chamberID)][1]
    tempDF["position"] = pos
    tempDF["chamber"] = chamberID
    tempDF["dose"] = avgCSV["dose"]
    return tempDF


def groupDataByFilter(setList, tgtPctT):
    fltrSetDF = pd.DataFrame()
    replicate = 1
    for i in range(len(setList)):
        avgStdDF = getData(chamberFolder, setList[i], replicate, tgtPctT)
        fltrSetDF = pd.concat([fltrSetDF, avgStdDF], axis=0)
        
        replicate+=1
    #print(fltrSetDF)
    fltrSetDF = fltrSetDF.sort_index()
    #(fltrSetDF)
    return fltrSetDF, replicate
    
def createFullDF(chFolder, chID, tgtTs, fullPropDF):
    for targetPctT in tgtTs:
        fileSetList = createFilterGroup(targetPctT, chFolder)
        filterSetDF, numberReps = groupDataByFilter(fileSetList, targetPctT)
        fullPropDF = pd.concat([fullPropDF, filterSetDF])
      
    return fullPropDF

# def plotTrendsCheck(fullDF):
#     fig, trendsX = plt.subplots(figsize=(6, 6), dpi=300)
#     chamberID = 5
    
#     plt.title(f"Normalized Area of Peak at {peakDict[targetPeak]} cm\u207b\u2071, {conditionsDict[chamberID][0]} \N{DEGREE SIGN}C/{conditionsDict[chamberID][1]}% RH")  
#     plt.xlabel(f"Dose (W/m\u00B2)")
#     plt.ylabel(f"Area ({peakDict[targetPeak]}) / Area ({normWn})")
#     #trendsX.axis([-5, 100, -0.1, 2.5])
#     chamberDF = fullDF[fullDF['chamber']==str(chamberID)]
#     #print(chamberDF)
#     for t in range(len(targetTs)):
#         filterDF = chamberDF[chamberDF['targetT']==targetTs[t]]
#         #print(filterDF)
#         positions = filterDF["position"].drop_duplicates().tolist()
#         for p in range(len(positions)):
#             subDF = filterDF[filterDF["position"]==positions[p]]
#             #print(subDF)
#             trendsX.plot(subDF.index, subDF["Avg"], linestyle= "none", color=colorList[t], markeredgecolor=colorList[t],markerfacecolor=colorList[t], marker='o', fillstyle=fillList[p], label = f"{subDF['pctT'].tolist()[0]} %T" )
#             (_, caps, _) = trendsX.errorbar(subDF.index, subDF["Avg"], subDF["Std"], capsize=5, c=colorList[t], fmt='none')
#             trendsX.legend(loc='upper left', bbox_to_anchor=(0.99, 0.99), frameon=False)

def plotTrends(fullDF, chamberID, figOutPath):
    
    fig, trendsX = plt.subplots(figsize=(8, 6), dpi=300, layout='constrained')
    #chamberID = 1
    #plt.title(f"Normalized Area of Peak at {peakDict[targetPeak]} cm\u207b\u00B9, {conditionsDict[chamberID][0]} \N{DEGREE SIGN}C/{conditionsDict[chamberID][1]}% RH", fontsize=22, y=1.05, family = fontString)  
    #plt.title(f"Ratio of Peak Area: {peakDict[targetPeak]} cm\u207b\u00B9/1685 cm\u207b\u00B9 \n {conditionsDict[chamberID][0]} \N{DEGREE SIGN}C/{conditionsDict[chamberID][1]}% RH \n (Thermal Control Subtracted)", fontsize=22, y=1.05, family = fontString) 
    dictKey = f"{trendType}{ratioVAbs}"
    print(dictKey)
    titleDict = {
        "AbsRatio": f"Peak Area Ratio: {peakDict[targetPeak]} cm\u207b\u00B9/{str(divisor)} cm\u207b\u00B9",
        "AbsCorrRatio": f"Peak Area Ratio: {peakDict[targetPeak]} cm\u207b\u00B9/{str(divisor)} cm\u207b\u00B9", 
        "DiffRatio": f"Change in Peak Area Ratio: {peakDict[targetPeak]} cm\u207b\u00B9/{str(divisor)} cm\u207b\u00B9",
        "DiffCorrRatio": f"Change in Peak Area Ratio: {peakDict[targetPeak]} cm\u207b\u00B9/{str(divisor)} cm\u207b\u00B9",
        "AbsAbs": f"Peak Area: {peakDict[targetPeak]} cm\u207b\u00B9",
        "AbsCorrAbs": f"Peak Are: {peakDict[targetPeak]} cm\u207b\u00B9", 
        "DiffAbs": f"Change in Peak Area Ratio: {peakDict[targetPeak]} cm\u207b\u00B9",
        "DiffCorrAbs": f"Change in Peak Area Ratio: {peakDict[targetPeak]} cm\u207b\u00B9"
        }
    titleStr = f"{titleDict[dictKey]}\n {conditionsDict[chamberID][0]} \N{DEGREE SIGN}C/{conditionsDict[chamberID][1]}% RH"
    
    #if "divided-Area" in targetProperty:
     #   titleStr = f"Divided {titleStr}/
    #plt.title(f"Ratio of Peak Area: {peakDict[targetPeak]} cm\u207b\u00B9/{str(divisor)} cm\u207b\u00B9 \n {conditionsDict[chamberID][0]} \N{DEGREE SIGN}C/{conditionsDict[chamberID][1]}% RH", fontsize=22, y=1.05, family = fontString)
    plt.title(titleStr, fontsize=22, y=1.05, family = fontString)
    maxX = max(fullDF['dose'])
    
    chamberDF = fullDF[fullDF['chamber']==str(chamberID)]
    trendsX.set_xlabel(f"Dose (W/m\u00B2)", fontsize=22, font = fontString)
    #trendsX.set_ylabel(f"Area ({peakDict[targetPeak]}) / Area ({normWn})", fontsize=22, font = fontString)
    trendsX.set_ylabel(f"\u0394 Product index ({peakDict[targetPeak]} cm\u207b\u00B9)", fontsize=22, font = fontString)
    trendsX.tick_params(axis='both', which='major', labelsize=22)
    plt.xticks(np.arange(min(fullDF['dose']), max(fullDF['dose'])+1, 50))
    plt.yticks(np.arange(round(min(fullDF['Avg']), 0), max(fullDF['Avg'])+5, 0.5))
    for tick in trendsX.get_xticklabels():
        tick.set_fontname(fontString)
    for tick in trendsX.get_yticklabels():
        tick.set_fontname(fontString)
    #trendsX.axis([-5, 100, -0.1, 2.5])
    
    #maxY = max(chamberDF['Avg'])
    #plt.yticks(np.arange(round(min(chamberDF['Avg']), 0), max(chamberDF['Avg'])+1, 0.5))
   
    #Main data plotting loop    
    for t in reversed(range(len(targetTs))):
        filterDF = chamberDF[chamberDF['targetT']==targetTs[t]]
        positions = filterDF["position"].drop_duplicates().tolist()
        for p in range(len(positions)):
            if positions[p]!=1 and positions[p]!=12:
                subDF = filterDF[filterDF["position"]==positions[p]]
                trendsX.plot(subDF['dose'], subDF["Avg"], linestyle= "-", color=colorList[t], markeredgecolor=colorList[t],markerfacecolor=colorList[t], marker='o', markersize=12, fillstyle=fillList[p], label = f"{subDF['pctT'].tolist()[0]} %T" )
                (_, caps, _) = trendsX.errorbar(subDF['dose'], subDF["Avg"], subDF["Std"], capsize=5, c=colorList[t], fmt='none')
     
    #setting y axis to leave room for thermal correction note             
    xmin, xmax, ymin, ymax = plt.axis()
    trendsX.set_ylim(top=ymax*1.1)
    #Thermal Correction Note
    if "Corr" in trendType: 
        plt.text(-5, ymax*1.03, f"Thermal Correction Applied", fontname = fontString, fontsize=18, fontstyle = 'italic')
    else: 
        plt.text(-5, ymax*1.03, f"No Thermal Correction", fontname = fontString, fontsize=18, fontstyle = 'italic')
    #add legend and save file
    trendsX.legend(loc='upper left', bbox_to_anchor=(0.99, 0.99), frameon=False, prop=dict(family=fontString, size=18))
    figFile = f"Chamber-{str(chamberID)}-peakTrend-{peakDict[targetPeak]}-allFilters.png"
    fig.savefig(figOutPath / figFile, format='png', dpi=300, transparent=True)


positionList = createPosList()
targetTs = filtersList()
fullPropertyDF = pd.DataFrame()
#for targetPctT in targetTs:
    #print(targetPctT)
for chamberFolder in trendsFTIRFolder.iterdir():
    chamberFigOutFolder = parentDir / folderNm / "x_peak-trends-by-T-figs" / str(normWn) / f"div-by-{divisor}" #/str(str(chamberFolder).split('\\')[-1:][0])
    chamberFigOutFolder.mkdir(parents=True, exist_ok=True)
    chamberID = str(chamberFolder).split("\\")[-1].split("-")[-1]
    fullPropertyDF = createFullDF(chamberFolder, chamberID, targetTs, fullPropertyDF)
    #dividedDF = divideByControl(fullPropertyDF, int(chamberID), positionList)
    #subtractedDF = subtractInitial(fullPropertyDF, int(chamberID), positionList)    
    #(fullPropertyDF, int(chamberID), chamberFigOutFolder)
    #plotTrends(dividedDF, int(chamberID), chamberFigOutFolder)
    plotTrends(fullPropertyDF, int(chamberID), chamberFigOutFolder)

