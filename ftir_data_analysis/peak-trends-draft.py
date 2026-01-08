
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

def getPosition(file):
    return int(re.sub('[a-zA-Z]+', '', file.split('-')[2]))

def getChamber(file):
    return int(re.sub('[a-zA-Z]+', '', file.split('-')[1]))


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
    byPosFolderNm = "4a_fit-by-position",
    avgStdFolderNm = "4b_fit-averages-stdevs",
    byPropFolderNm = "5a_fit-avg-std_by-prop_no-corr",
    normWn = '723',
    # numMeas = 4         #to-do: get away from hard coding this 
    ):

    rawDataFolder = Path(rawDataDir)
    parentDir = rawDataFolder.absolute().parent 
    byPosFolder = parentDir / byPosFolderNm / normWn
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
            dosesArray = np.loadtxt(dosesCSVPath, delimiter=',', usecols=range(4,22), dtype=str)
            position = re.sub('[a-zA-Z]+', '', str(Path(root).parent).split('\\')[-1].split('-')[-1])
            dosesList = [float(dosesArray[np.where(dosesArray[:, 0] == f"{int(expHr)}h")[0][0], np.where(dosesArray[0, :] == str(position))[0][0]]) for expHr in expHrsList]

            #local function for positionArray list comprehension
            def importDataFunc(file):
                return np.loadtxt(Path(root) / file, delimiter=',', skiprows=1)
            positionArray = np.array([importDataFunc(filename) for filename in files])

            #create arrays for each property 
            for column in range(0, positionArray.shape[2]):
                propertyArray = np.column_stack((expHrsList, dosesList, positionArray[:, :, column])) 
                #making file folders
                propValueFolder = byPropOutFolder / propertyNames[column] / root.split('\\')[-1]
                propValueFolder.mkdir(parents=True, exist_ok=True)

                # #making file names
                sourceFilename = files[0]
                sample = sourceFilename.split('_')[1].replace('-0h', '')
                valueType = sourceFilename.split('_')[-1].split('.')[0]
                outFileName = f"{sample}_N{normWn}_{propertyNames[column]}_{valueType}.csv"
                
                #save files
                fmtList = ['%f' for wn in avgWns]
                fmtList.insert(0,'%f')
                fmtList.insert(0,'%i')
                np.savetxt(propValueFolder / outFileName, propertyArray, delimiter=',', header=f"expHrs,dose,{wnString}", comments='', fmt=fmtList)  #adds wavenumbers as column headers



#run after arrange to perform math on fit arrays 
def subtractInitial(
    rawDataDir,
    byPosFolderNm = "4a_fit-by-position",
    byPropFolderNm = "5a_fit-avg-std_by-prop_no-corr",
    initSubFolderNm = "5b_fit-delCI_by-prop_no-corr",
    normWn = '723',
    xCols = 2                                                   #number of columns containing x data (eg exposure hours and dose)
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
                
                #local function for subtraction for averages 
                def subtractInitStDev(row):
                    n = numReps(row)
                    return np.sqrt(np.add(np.divide(np.power(propertyArray[row,xCols:], 2), n), np.divide(np.power(propertyArray[row,xCols:], 2), n)))  #calculated for difference of means - sqrt((sd1^2/n1) + (sd2^2/n2))  

                if root.split('\\')[-1] == 'Average':
                    resultsArray = np.column_stack((propertyArray[:, :xCols], np.array([subtractInitAvg(rowNum) for rowNum in range(propertyArray.shape[0])]))) 
                if root.split('\\')[-1] == 'StDev':
                    resultsArray = np.column_stack((propertyArray[:, :xCols], np.array([subtractInitStDev(rowNum) for rowNum in range(propertyArray.shape[0])])))

                # print(f"resultsArray.shape: {resultsArray.shape}")

                #making file folders and file name
                subResultFolder = Path(root.replace(byPropFolderNm, initSubFolderNm))
                subResultFolder.mkdir(parents=True, exist_ok=True)
                subResultFileNm = file.replace(file.split('_')[-1], f"initSub_{file.split('_')[-1]}")

                fmtList = ['%f' if colNames.index(name)>=1 else '%i'for name in colNames]
                np.savetxt(subResultFolder / subResultFileNm, resultsArray, delimiter=',', header=','.join(colNames), comments='', fmt=fmtList)  #adds wavenumbers as column headers





if __name__ == "__main__":          #does not run if importing only if running
    # arrange("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-exposure/0_raw-data") 
    subtractInitial("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-exposure/0_raw-data") 