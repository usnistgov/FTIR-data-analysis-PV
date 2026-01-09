
# -*- coding: utf-8 -*-
"""

@author: klj
"""

import os, itertools  #os needed for os.walk. itertools allows for listcomprehension, speficially itertools.groupby allows easy grouping of replicate files 
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil                       #to delete temporary folder for paired differences 
import re #regular expressions for list comprehension and string parsing 

def getPosition(file):
    return int(re.sub('[a-zA-Z]+', '', file.split('-')[2]))

def getChamber(file):
    return int(re.sub('[a-zA-Z]+', '', file.split('-')[1]))

def getExpHours(file):
    return int(re.sub('[a-zA-Z]+', '', file.split('-')[3]))


#get average wavenumbers for the entire data set to use for column headers, for uniformity, readability, and easier data matching
def averageWavenumbers(folder):
    #local function for list comprehension
    def importDataFunc(file):
        return np.loadtxt(Path(root) / file, delimiter=',', skiprows=1, usecols=0)
    
    for root, dirs, files in os.walk(folder):   #loop through full folder 
            if len(files)>0 and root.split('\\')[-1] == 'Average':   #if files are in folder and only average folders 
                wnArray = np.array([importDataFunc(filename) for filename in files])    #uses local load function to get first column of each file (wavenumbers) and creates array of all of them, each file as new row
                wnArrayFull = wnArray if 'wnArrayFull' not in locals() else np.concatenate((wnArrayFull, wnArray), axis=0)  #stacks arrays together 
    return wnArrayFull.mean(axis=0).astype(int).astype(str).tolist()    #list of average values for wavenumbers, converted to integers then strings

def arrange(
    rawDataDir,
    avgStdFolderNm = "4b_fit-averages-stdevs",
    byPropFolderNm = "5a_fit-AvgStd_byProp_noCorr",
    normWn = '723',
    pairedDiffData = False
    ):

    rawDataFolder = Path(rawDataDir)
    parentDir = rawDataFolder.absolute().parent 
    avgStdFolder = parentDir / avgStdFolderNm / normWn
    byPropFolder = parentDir / byPropFolderNm / normWn

    #get average wavenumbers for the entire data set to use for column headers, for uniformity, readability, and easier data matching
    avgWns = averageWavenumbers(avgStdFolder)
    wnString = ','.join(avgWns)

    for root, dirs, files in os.walk(avgStdFolder):
        
        if len(files)>0:
            byPropOutFolder = Path(str(root).replace(str(avgStdFolder), str(byPropFolder))).parent.parent #output folder for avg and std dev 
            propertyNames = np.genfromtxt(Path(root) / files[0], delimiter=',', names=True, max_rows=1).dtype.names #retrieving column names. genfromtxt stores column names as dtype. 
            expHrsList = [float(re.sub('[a-zA-Z]+', '', file.split('-')[3])) for file in files] #specific to my file naming conventions! grabs section with exposure hours and removes all alphanumberics (regex pattern '[a-zA-Z]+')

            chID = str(Path(root).parent.parent).split('\\')[-1].split('-')[-1]
            dosesCSVPath = parentDir.parent / f"Doses-Ch-{chID}.csv"
            dosesArray = np.loadtxt(dosesCSVPath, delimiter=',', usecols=range(3,22), dtype=str)
            position = re.sub('[a-zA-Z]+', '', str(Path(root).parent).split('\\')[-1].split('-')[-1])
            #adding doses and actual exposure times as optional x columns
            dosesList = [float(dosesArray[np.where(dosesArray[:, 1] == f"{int(expHr)}h")[0][0], np.where(dosesArray[0, :] == str(position))[0][0]]) for expHr in expHrsList]
            actExpHrList = [float(dosesArray[np.where(dosesArray[:, 1] == f"{int(expHr)}h")[0][0], 0]) for expHr in expHrsList]

            #local function for positionArray list comprehension
            def importDataFunc(file):
                return np.loadtxt(Path(root) / file, delimiter=',', skiprows=1)
            positionArray = np.array([importDataFunc(filename) for filename in files])

            #create arrays for each property 
            for column in range(0, positionArray.shape[2]):
                propertyArray = np.column_stack((expHrsList, actExpHrList, dosesList, positionArray[:, :, column])) 
                #making file folders
                propValueFolder = byPropOutFolder / propertyNames[column] / root.split('\\')[-1]
                propValueFolder.mkdir(parents=True, exist_ok=True)

                # #making file names
                sourceFilename = files[0]
                sample = sourceFilename.split('_')[1].replace('-0h', '')
                valueType = sourceFilename.split('_')[-1].split('.')[0]
                outFileName = f"{sample}_N{normWn}_{propertyNames[column]}_{valueType}.csv" if pairedDiffData==False else f"{sample}_N{normWn}_{propertyNames[column]}_pairedInitSub_{valueType}.csv"

                #save files
                fmtList = ['%f' for wn in avgWns]
                fmtList.insert(0,'%f')  #for dose - format float 
                fmtList.insert(0,'%f')  #for actual exp hr - format float 
                fmtList.insert(0,'%i')  #for rounded exp hr - format int 
                np.savetxt(propValueFolder / outFileName, propertyArray, delimiter=',', header=f"expHrs,actExpHrs,dose,{wnString}", comments='', fmt=fmtList)  #adds wavenumbers as column headers

#run after arrange to perform math on fit arrays 
def subtractInitialUnpaired(
    rawDataDir,
    byPosFolderNm = "4a_fit-by-position",
    byPropFolderNm = "5a_fit-AvgStd_byProp_noCorr",
    initSubFolderNm = "5bi_fit-delCI-unpaired_byProp_noCorr",
    normWn = '723',
    xCols = 3                                                   #number of columns containing x data (eg exposure hours and dose)
    ):
    rawDataFolder = Path(rawDataDir)
    parentDir = rawDataFolder.absolute().parent 
    byPosFolder = parentDir / byPosFolderNm / normWn
    byPropFolder = parentDir / byPropFolderNm / normWn
    initSubFolder =  parentDir / initSubFolderNm / normWn

    for root, dirs, files in os.walk(byPropFolder):
        if len(files)>0:
            # initSubOutFolder = Path(str(root).replace(str(byPropFolder), str(initSubFolder))).parent #output folder for avg and std dev 
            for file in files:
                byPosPath = byPosFolder / f"chamber-{getChamber(file)}" / f"Pos{getPosition(file)}" #corresponding path containing the original replicate data pre-averaging 
                colNames = np.genfromtxt(Path(root)/file, delimiter=',', names=True, max_rows=1).dtype.names       #keep column names to add back in when saving
                propertyArray = np.loadtxt(Path(root)/file, delimiter=',', skiprows=1)
                
                #local function for finding the number of repliates, needed for calculating standard deviation of difference of means  
                def numReps(row):   
                    expHrs = int(propertyArray[row, colNames.index('expHrs')])
                    searchTerm = file.split('_')[0].split('-')
                    searchTerm.insert(-1, f"{expHrs}h")
                    searchTerm = '-'.join(searchTerm)
                    replicateList = [filename for filename in byPosPath.iterdir() if searchTerm in str(filename)]
                    return len(replicateList)

                #local function for subtraction for averages 
                def subtractInitAvg(row):
                    return np.subtract(propertyArray[row,xCols:], propertyArray[0,xCols:])      #only performs math on non-x colulmns - x columns will be added back in  
                
                #local function for standard deviation 
                def subtractInitStDev(row):
                    n = numReps(row)
                    return np.sqrt(np.add(np.divide(np.power(propertyArray[row,xCols:], 2), n), np.divide(np.power(propertyArray[row,xCols:], 2), n)))  #calculated for difference of means - sqrt((sd1^2/n1) + (sd2^2/n2))  

                if root.split('\\')[-1] == 'Average':
                    resultsArray = np.column_stack((propertyArray[:, :xCols], np.array([subtractInitAvg(rowNum) for rowNum in range(propertyArray.shape[0])]))) 
                if root.split('\\')[-1] == 'StDev':
                    resultsArray = np.column_stack((propertyArray[:, :xCols], np.array([subtractInitStDev(rowNum) for rowNum in range(propertyArray.shape[0])])))

                #making file folders and file name
                subResultFolder = Path(root.replace(byPropFolderNm, initSubFolderNm))
                subResultFolder.mkdir(parents=True, exist_ok=True)
                subResultFileNm = file.replace(file.split('_')[-1], f"initSub_{file.split('_')[-1]}")

                fmtList = ['%f' if colNames.index(name)>=1 else '%i'for name in colNames]
                np.savetxt(subResultFolder / subResultFileNm, resultsArray, delimiter=',', header=','.join(colNames), comments='', fmt=fmtList)  #adds wavenumbers as column headers


#creates averages and standard deviations for replicates sets and saves files. 
#Uses paired data analysis methods and pairs replicates 1,2,3,4 in exposed sample with replicates 1,2,3,4 in same unexposed sample. 
def subtractInitialPaired(
    rawDataDir,
    byPosFolderNm = "4a_fit-by-position",
    pairedDiffFolderNm = "4c_fit-pairedDiffs-avg-std-by-pos",
    byPropFolderNm = "5bii_fit-delCI-paired_byProp_noCorr",
    normWn = '723',
    ):

    rawDataFolder = Path(rawDataDir)
    parentDir = rawDataFolder.absolute().parent 
    byPosFolder = parentDir / byPosFolderNm / normWn
    pairedDiffFolder = parentDir / pairedDiffFolderNm / normWn

    for root, dirs, files in os.walk(byPosFolder):
        # depth = root.replace(str(byPosFolder), '').count(os.sep)
        if len(files)>0:
            pairedDiffPosFolder = Path(str(root).replace(str(byPosFolder), str(pairedDiffFolder))) #output folder for paired avg and std dev 
            fileSets = [list(g) for _, g in itertools.groupby(files, lambda x: x[:x.rfind('_')])]     #groups file replicates in list 
            
        #creates 3d arrays for each file set. 3d arrays are sets of fit results for sets of replicates.
            columnNames = ','.join(np.genfromtxt(Path(root) / fileSets[0][0], delimiter=',', names=True, max_rows=1).dtype.names) #retrieving column names. genfromtxt stores column names as dtype. join combines list into string with delimiter ','
            #local function for setArray list comprehension
            def importDataFunc(file):
                return np.loadtxt(Path(root) / file, delimiter=',', skiprows=1)
            
            for set in fileSets:
                if getExpHours(set[0])==0:
                    initSetArray = np.array([importDataFunc(filename) for filename in set]) # for set in fileSets if getExpHours(set[0])==0]
                    
            
            for set in fileSets:  
                #creates 3d arrays for each file set. 3d arrays are sets of fit results for sets of replicates.
                setArray = np.array([importDataFunc(filename) for filename in set])
                #creates 3d arrays for each file set. 3d arrays are sets of results from subtracting the t!=0 array elements from the t=0 array elements, in other words, rep 1 (t!=0) - rep 1 (t=0) and so on.
                pairedDiffArray = np.array([np.column_stack((setArray[array, :, 0], np.subtract(setArray[array, :, 1:], initSetArray[array, :, 1:]))) for array in range(setArray.shape[0])])
                #creates 2d array from averages of these differences. 
                pairedDiffAvgArray = pairedDiffArray[:, :, :].mean(axis=0)  
                #creates a 2d array from standard deviations of these differences, using the wavenumber column from the average data. 
                pairedDiffStDevArray = np.column_stack((pairedDiffAvgArray[:, 0], pairedDiffArray[:, :, 1:].std(axis=0, ddof=1)))   #ddof is delta degrees of freedom. divisor n -ddof. using 1 since this is a sample not a population. https://numpy.org/devdocs/reference/generated/numpy.std.html 
                #creating file names and folders
                filePrefix = set[0][:(set[0].rfind('_'))]
                avgPosFolder, stdPosFolder = pairedDiffPosFolder / 'Average', pairedDiffPosFolder / 'StDev'
                avgPosFolder.mkdir(parents=True, exist_ok=True)
                stdPosFolder.mkdir(parents=True, exist_ok=True)
                #output data to folder 
                np.savetxt(avgPosFolder / f"{filePrefix}_Avg.csv", pairedDiffAvgArray, delimiter=',', header=columnNames, comments='')
                np.savetxt(stdPosFolder / f"{filePrefix}_StDev.csv", pairedDiffStDevArray, delimiter=',', header=columnNames, comments='')
    #these are of the same format as the basic average and standard deviations arranged by position and need to be put through arrange function to be formatted for plotting over time/exposure. 
    arrange(rawDataDir, avgStdFolderNm = pairedDiffFolderNm, byPropFolderNm = byPropFolderNm, normWn = normWn, pairedDiffData=True)

#run after arrange to perform math on fit arrays 
def subtractDarkControl(
    rawDataDir,
    paired = True,
    byPosFolderNm = "4a_fit-by-position", #needed to get number of measurements for difference of means 
    initSubFolderNm = "5bii_fit-delCI-paired_byProp_noCorr",
    darkCorrFolderNm = "5cii_fit-delCI-paired_byProp_darkCorr",
    normWn = '723',
    xCols = 3                                                   #number of columns containing x data (eg exposure hours and dose)
    ):
    rawDataFolder = Path(rawDataDir)
    parentDir = rawDataFolder.absolute().parent 
    byPosFolder = parentDir / byPosFolderNm / normWn
    initSubFolder =  parentDir / initSubFolderNm / normWn

    #for file counting only  
    totalAvg, totalStDev = 0, 0
    avgCounter, stDevCounter = 0, 0
    for root, dirs, files in os.walk(initSubFolder):
        for file in files: 
            if root.split('\\')[-1] == 'Average':
                totalAvg +=1
            if root.split('\\')[-1] == 'StDev':
                totalStDev +=1
    print(f'\n total file sets for correcting: {totalAvg} Avgs and {totalStDev} StDevs')
    
    for root, dirs, files in os.walk(initSubFolder):
        if len(files)>0:    
            for file in files: 
                darkFile = file if getPosition(file)==1 else None   #find file at position 1 (dark control file)
                break
            for file in files:
                 
                colNames = np.genfromtxt(Path(root)/file, delimiter=',', names=True, max_rows=1).dtype.names       #keep column names to add back in when saving
                propertyArray = np.loadtxt(Path(root)/file, delimiter=',', skiprows=1)
                darkPropArray = np.loadtxt(Path(root)/darkFile, delimiter=',', skiprows=1)
                
                #local function for finding the number of replicates, needed for calculating standard deviation of difference of means  
                def numReps(expHrs, searchFile):   
                    byPosPath = byPosFolder / f"chamber-{getChamber(searchFile)}" / f"Pos{getPosition(searchFile)}" #corresponding path containing the original replicate data pre-averaging
                    searchTerm = searchFile.split('_')[0].split('-')
                    searchTerm.insert(-1, f"{int(expHrs)}h")
                    searchTerm = '-'.join(searchTerm)
                    replicateList = [filename for filename in byPosPath.iterdir() if searchTerm in str(filename)]
                    return len(replicateList)
            
                #local function for subtraction for averages 
                def subtractDarkAvg(row):
                    expHr = propertyArray[row, colNames.index('expHrs')]
                    darkRow = np.where(darkPropArray[:, colNames.index('expHrs')] == expHr)[0][0]        #matches expHr to expHr  - position 1 has all exposure times but not all positions are measured at every time
                    return np.subtract(propertyArray[row, xCols:], darkPropArray[darkRow, xCols:])       #only performs math on non-x colulmns - x columns will be added back in   
                
                #local function for subtraction for averages 
                def subtractDarkStDev(row):
                    expHr = propertyArray[row, colNames.index('expHrs')]
                    darkRow = np.where(darkPropArray[:, colNames.index('expHrs')] == expHr)[0][0]        #matches expHr to expHr  - position 1 has all exposure times but not all positions are measured at every time    
                    n, darkn = numReps(expHr, file), numReps(expHr, darkFile)
                    return np.sqrt(np.add(np.divide(np.power(propertyArray[row,xCols:], 2), n), np.divide(np.power(darkPropArray[darkRow,xCols:], 2), darkn)))  #calculated for difference of means - sqrt((sd1^2/n1) + (sd2^2/n2))  
                
                #perform operations to achieve average and standard deviations for dark correction
                if root.split('\\')[-1] == 'Average':
                    resultsArray = np.column_stack((propertyArray[:, :xCols], np.array([subtractDarkAvg(rowNum) for rowNum in range(propertyArray.shape[0])])))
                    avgCounter+=1
                    

                if root.split('\\')[-1] == 'StDev':
                    resultsArray = np.column_stack((propertyArray[:, :xCols], np.array([subtractDarkStDev(rowNum) for rowNum in range(propertyArray.shape[0])])))
                    stDevCounter+=1

                #making file folders and file name
                darkCorrResultFolder = Path(root.replace(initSubFolderNm, darkCorrFolderNm))
                darkCorrResultFolder.mkdir(parents=True, exist_ok=True)
                darkCorrResultFileNm = file.replace("_".join(file.split('_')[-2:]), f"darkCorr_{file.split('_')[-1]}") if paired==False else file.replace("_".join(file.split('_')[-2:]), f"pairedDarkCorr_{file.split('_')[-1]}")
            
                fmtList = ['%f' if colNames.index(name)>=1 else '%i'for name in colNames]
                np.savetxt(darkCorrResultFolder / darkCorrResultFileNm, resultsArray, delimiter=',', header=','.join(colNames), comments='', fmt=fmtList)  #adds wavenumbers as column headers

                print(f'\r completed {str(avgCounter)}/{str(totalAvg)} avg files and {str(stDevCounter)}/{str(totalStDev)} st dev files, {str(round(((avgCounter+stDevCounter)/(totalAvg+totalStDev) *100), 1))}% complete', end=' ')    #prints on one line


if __name__ == "__main__":          #does not run if importing only if running
    # arrange("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-exposure/0_raw-data") 
    # subtractInitialUnpaired("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-exposure/0_raw-data") 
    # subtractInitialPaired("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-exposure/0_raw-data")  
    subtractDarkControl("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-exposure/0_raw-data", paired=True, initSubFolderNm = "5bii_fit-delCI-paired_byProp_noCorr", darkCorrFolderNm = "5cii_fit-delCI-paired_byProp_darkCorr")
    subtractDarkControl("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-exposure/0_raw-data", paired=False, initSubFolderNm = "5bi_fit-delCI-unpaired_byProp_noCorr", darkCorrFolderNm = "5ci_fit-delCI-unpaired_byProp_darkCorr")