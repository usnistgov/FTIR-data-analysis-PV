
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:12:48 2025

@author: klj
"""

##currently not very good and pretty messy. right now, this groups properties of interest for each position
##given a property of interest (input as function param for propsDF)

# it would be good to be able to plot grouped by %T filter.

import os 
from pathlib import Path
import numpy as np
import pandas as pd
#from scipy.signal import find_peaks as find_peaks
#from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math

np.set_printoptions(suppress=True)
folderNm='FTIR-data-PET-exposure'  
parentDir = Path().absolute().parent
normWn = '723'
fitFTIRFolder = parentDir / folderNm / "3_fit-results" / normWn
numMeas = 4

peakDict = {0: "631", 1: "652", 2: "689", 3: "711", 4: "725", 5: "1578", 6: "1610", 7: "1650", 8: "1685", 9: "1714", 10: "1727", 11: "1735", 12: "1749", 13: "1777", 14: "1857"}

colorList = ["#000000", "#FF0000", "#FF8000", "#00FF00", "#009300", "#00C8C8"]

#creates list of all positions for which there is data in the folder. 
def createPosList(chFolder):
    posList = []
    for dateFolder in chFolder.iterdir():
        for filename in os.listdir(dateFolder):
            if filename[-7:-4] != "Avg" and filename.split("-")[4].split("_")[0]=="Air":
                posNum = int(filename.split("-")[2].replace("Pos", "").replace("pos", ""))
                if posNum not in posList: 
                    posList.append(posNum)
    posList.sort()
    return posList

#makes a list of file names (except meas number and extension) containing data for each position.
def findPositionFiles(pos, chFolder):
    fileList = []
    for dateFolder in chFolder.iterdir():
        for filename in os.listdir(dateFolder):
            filePos = filename.split("-")[2].replace("Pos", "").replace("pos", "")  #collects position number from file name 
            airBack = filename.split("-")[4].split("_")[0]                          #collects air v back designation from file name
            expHours = filename.split("-")[3].split("_")[0].replace("h", "").replace("H", "").replace("hr", "").replace("Hr", "")  
            if filePos == str(pos) and filename[-7:-4] != "Avg" and airBack=="Air": #currently this script ONLY ACCESSES AIR. this will need to be addressed if back side data are desired.                 
                fileSet = filename[:-6]
                if fileSet not in fileList:
                    fileList.append(fileSet)                                        #if the file matches the position being collected, the filename is added to the list for future file access. 
                if expHours == "0":
                    zeroHoursFileSet = fileSet
                    zeroHoursDateFolder = dateFolder
    return fileList, zeroHoursFileSet, zeroHoursDateFolder

#obtains a CSV for the input folders and filename, as well as the number of peak properties listed in the table
#and the hours of exposure. appends all replicates together into one large DF. 
def getData(chFolder, dateFolder, fileSetNm):
    expHours = fileSetNm.split("-")[3].replace("h", "").replace("H", "").replace("hr", "").replace("Hr", "")
    fullCSV = pd.DataFrame()
    for i in range (1, numMeas+1):
        tempCSV = pd.read_csv(chFolder / dateFolder/ f"{fileSetNm}_{i}.csv", sep=',', header=0, on_bad_lines="skip", engine='python')
        tempCSV = pd.read_csv(chFolder / dateFolder/ f"{fileSetNm}_{i}.csv", sep=',', header=0, on_bad_lines="skip", engine='python')
        fullCSV = pd.concat([fullCSV, tempCSV], axis=1)
    numProperties = len(tempCSV.columns)
    return fullCSV, expHours, numProperties



#averages together sets of 4 measurements and creates an averages and standard deviation DF. 
def avgAbsSets(df, absVsDiff):
    avgsDF = pd.DataFrame()
    stdDF = pd.DataFrame()
    skipInt = int(len(df.columns)/numMeas) #effectively the number of columns for each measurement
    for i in range(skipInt):
        #print(i)
        forAvgDF = df.iloc[:,i::skipInt]
        #print(forAvgDF)
        columnNm = forAvgDF.columns[0]
        avgsDF[columnNm] = forAvgDF.mean(axis=1)
        stdDF[columnNm] = forAvgDF.std(axis=1) #does do standard deviation of wavenumbers 
        # if i==0: 
        #     stdDF[columnNm] = forAvgDF.mean(axis=1) #does not do standard deviation of wavenumbers
        # else:    
        #     stdDF[columnNm] = forAvgDF.std(axis=1)
        #print(avgsDF)
    return avgsDF, stdDF

#averages differences (both for difference from t=0 and subtracting the dark correction)
def avgDiffSets(dfA, stdDfA, dfB, stdDfB):
    #print(stdDfB)
    expHr = str(dfA.columns[0].split(" ")[-1])
    for column in dfA.columns:
        dfA.rename(columns={column: column.replace(f" {expHr}", "")}, inplace=True)
        stdDfA.rename(columns={column: column.replace(f" {expHr}", "")}, inplace=True)
        dfB.rename(columns={column: column.replace(f" {expHr}", "")}, inplace=True)
        stdDfB.rename(columns={column: column.replace(f" {expHr}", "")}, inplace=True)
    diffDF = dfA.sub(dfB)
    diffDF.iloc[:,0]= dfA.iloc[:,0]

    diffStdDevDF = np.sqrt((stdDfA.astype('float64').pow(2)).div(numMeas).add((stdDfB.astype('float64').pow(2)).div(numMeas))) #calculated for difference of means - sqrt((sd1^2/n1) + (sd2^2/n2))
    diffStdDevDF.iloc[:,0]= dfA.iloc[:,0]
    
    for column in diffDF.columns:
        diffDF.rename(columns={column: f"{column} {expHr}"}, inplace=True)
        diffStdDevDF.rename(columns={column: f"{column} {expHr}"}, inplace=True)
    return diffDF, diffStdDevDF

#adds data onto existing DFs by adding columns. Adds the exposure hours to the column name
#also adds the dose as a new row at the bottom.
def appendDF(runningDF, newDF, hrVal, doseVal):
    for column in newDF.columns:
        newDF.rename(columns={column: column.replace(f" {hrVal}", "")}, inplace=True)
    newRow = pd.DataFrame(columns=newDF.columns, index=['dose'])
    #print(newDF)
    for column in newDF.columns:
        newDF.rename(columns={column: f"{column} {hrVal}"}, inplace=True)
        newRow.rename(columns={column: f"{column} {hrVal}"}, inplace=True)
        newRow.at['dose', f"{column} {hrVal}"]=doseVal
    #print(newRow)
    newDF = pd.concat([newDF, newRow], axis=0)
    runningDF = pd.concat([runningDF, newDF], axis=1)
    #print(runningDF)
    return runningDF

def createAvgandDiffDFs(pos, chFolder):
    #finding all the files for position (chamber specific)
    posFileList, initFile, zeroHrDateFolder = findPositionFiles(pos, chFolder)
    initSetDFAll, expHr, numProp = getData(chFolder, zeroHrDateFolder, initFile)  
    initSetAvgDF, initSetStdDF = avgAbsSets(initSetDFAll, "absolute")
    #create absDFs - empty at first, but to contain the absolute values of the parameters in the files
    #one for averages and one for standard deviations 
    absAvgsDF = pd.DataFrame()
    absStDevDF = pd.DataFrame()
    #one for averages and one for standard deviations 
    diffAvgsDF = pd.DataFrame()
    diffStDevDF = pd.DataFrame()
    for fileSet in posFileList:
        for dateFolder in chFolder.iterdir():  
            if f"{fileSet}_1.csv" in os.listdir(dateFolder): #some folders won't have a specific position so this skips those
                timePointDF, expHr, _ = getData(chFolder, dateFolder, fileSet)
                dose = doseCSV[str(pos)][f"{expHr}h"]
                #absolute values for one file set (set of 4 replicates)
                indAbsAvgsDF, indAbsStdDF = avgAbsSets(timePointDF, "absolute")
                #append to running DF for positions
                absAvgsDF = appendDF(absAvgsDF, indAbsAvgsDF, expHr, dose) #combines with DFs for other file sets in the same position, by adding columms. 
                absStDevDF = appendDF(absStDevDF, indAbsStdDF, expHr, dose)
                #difference values (for both absolute areas and for all ratios columns)
                indDiffAvgsDF, indDiffStdDF = avgDiffSets(indAbsAvgsDF, indAbsStdDF, initSetAvgDF, initSetStdDF)
                # #append to running DF for position 
                diffAvgsDF = appendDF(diffAvgsDF, indDiffAvgsDF, expHr, dose)
                diffStDevDF = appendDF(diffStDevDF, indDiffStdDF, expHr, dose)                  
    return absAvgsDF, absStDevDF, diffAvgsDF, diffStDevDF, numProp


#accepts: full dataframe containing all time points for position as generated by appendDF
#breaks down into component time points again for subtraction 
def createTimePointDF(hrVal, fullDF):
        tmPtDF = pd.DataFrame()
        for column in fullDF.columns:
            if column.split(" ")[-1] == hrVal:
                tmPtDF = pd.concat([tmPtDF, fullDF[column]], axis=1)
        return tmPtDF

def subtractDarkDF(avgsDF, stdDF, dkAvgsDF, dkStdDF):
    #print(avgsDF.columns)
    expHrList = []
    avgsCorrDF = pd.DataFrame()
    stdCorrDF = pd.DataFrame()
    #gather all exposure hours into list 
    for column in avgsDF.columns:
        expHrList.append(column.split(" ")[-1])
    expHrList = list(set(expHrList))
    expHrList.sort()
    #for each exposure hour, recreate the time point data frame subset from the new averaged and subtracted
    #dataframes. subtract the corresponding dark value.    
    for expHr in expHrList:
        
        timePointDF = createTimePointDF(expHr, avgsDF)
        timePtStdDF = createTimePointDF(expHr, stdDF)
        dkAvgTPDF = dkAvgsDF[timePointDF.columns].copy()
        dkStdTPDF = dkStdDF[timePtStdDF.columns].copy()
        
        indAvgsCorrDF, indStdCorrDF = avgDiffSets(timePointDF, timePtStdDF, dkAvgTPDF, dkStdTPDF)
        
        avgsCorrDF = pd.concat([avgsCorrDF, indAvgsCorrDF], axis=1)
        stdCorrDF = pd.concat([stdCorrDF, indStdCorrDF], axis=1)
    #print(avgsDF.columns)
    for column in avgsCorrDF.columns:
        avgsCorrDF.at['dose', column] = avgsDF.at['dose', column]
        stdCorrDF.at['dose', column] = avgsDF.at['dose', column]

    return avgsCorrDF, stdCorrDF

def propsDFs(targetProp, allPropsDF, allPropsStdDF, numProps, chOutFolder, absVsDiff):
    #coreColList = allPropsDF.columns.tolist()[:numProps]
    for i in range(numProps):
        propertyNm = "-".join(allPropsDF.columns.tolist()[i].split(" ")[:-1]).replace("(", "").replace(")", "")
        #print(propertyNm)
        if targetProp in propertyNm:
            #print(propertyNm)
            propertyDF = allPropsDF.iloc[:,i::numProps]
            propertyStdDF = allPropsStdDF.iloc[:,i::numProps]
            for column in propertyDF:
                propertyDF.rename(columns={column: column.split(" ")[-1]}, inplace=True)
                propertyStdDF.rename(columns={column: column.split(" ")[-1]}, inplace=True)
            propertyDF = propertyDF.transpose()
            propertyStdDF = propertyStdDF.transpose()
            #print(propertyDF)
            avgFileNm = f"PET-Ch-{chamberID}-Pos{position}-Air-{propertyNm}-{absVsDiff}-Avg.csv"
            stdFileNm = f"PET-Ch-{chamberID}-Pos{position}-Air-{propertyNm}-{absVsDiff}-Std.csv"
            outputPath = chOutFolder / propertyNm / absVsDiff
            #print(outputPath)
            outputPath.mkdir(parents=True, exist_ok=True)
            propertyDF.to_csv(outputPath / avgFileNm, index=True)
            propertyStdDF.to_csv(outputPath / stdFileNm, index=True)

for chamberFolder in fitFTIRFolder.iterdir():
    #finding doses CSV to get doses from exposure hours 
    chamberID = str(chamberFolder).split("\\")[-1].split("-")[-1]
    doseCSV = pd.read_csv(parentDir / f"Doses-Ch-{chamberID}.csv", sep=',', header=0, index_col="file suffix" , on_bad_lines="skip", engine='python')
    #creating output Folder
    chamberDataOutFolder = parentDir / folderNm / "4_peak-trends" / str(normWn) /str(str(chamberFolder).split('\\')[-1:][0])
    chamberDataOutFolder.mkdir(parents=True, exist_ok=True)
    positionList = createPosList(chamberFolder)
    #create absolute values and differences from t=0 for position 1 (dark)
    p1_absAvgsDF, p1_absStDevDF, p1_diffAvgsDF, p1_diffStDevDF, _ = createAvgandDiffDFs(1, chamberFolder)

    
    for position in positionList:  
        #create absolute values and differences from t=0 for all positions
        posAbsAvgsDF, posAbsStDevDF, posDiffAvgsDF, posDiffStDevDF, numProperties = createAvgandDiffDFs(position, chamberFolder)

        posAbsAvgsCorrDF, posAbsStdCorrDF  = subtractDarkDF(posAbsAvgsDF, posAbsStDevDF, p1_absAvgsDF, p1_absStDevDF)
        posDiffAvgsCorrDF, posDiffStdCorrDF  = subtractDarkDF(posDiffAvgsDF, posDiffStDevDF, p1_diffAvgsDF, p1_diffStDevDF)

        propsDFs(f"area-N{normWn}", posAbsAvgsDF, posAbsStDevDF, numProperties, chamberDataOutFolder, "Abs")
        propsDFs(f"area-N{normWn}", posAbsAvgsCorrDF, posAbsStdCorrDF, numProperties, chamberDataOutFolder, "AbsCorr")
        propsDFs(f"area-N{normWn}", posDiffAvgsDF, posDiffStDevDF, numProperties, chamberDataOutFolder, "Diff")
        propsDFs(f"area-N{normWn}", posDiffAvgsCorrDF, posDiffStdCorrDF, numProperties, chamberDataOutFolder, "DiffCorr")
        
        for ind in range(4,15):
            propsDFs("divided-Area-by"+str(peakDict[ind]), posAbsAvgsDF, posAbsStDevDF, numProperties, chamberDataOutFolder, "Abs")
            propsDFs("divided-Area-by"+str(peakDict[ind]), posAbsAvgsCorrDF, posAbsStdCorrDF, numProperties, chamberDataOutFolder, "AbsCorr")
            propsDFs("divided-Area-by"+str(peakDict[ind]), posDiffAvgsDF, posDiffStDevDF, numProperties, chamberDataOutFolder, "Diff")
            propsDFs("divided-Area-by"+str(peakDict[ind]), posDiffAvgsCorrDF, posDiffStdCorrDF, numProperties, chamberDataOutFolder, "DiffCorr")
            propsDFs("wavenumbers", posAbsAvgsDF, posAbsStDevDF, numProperties, chamberDataOutFolder, "Wns")

    