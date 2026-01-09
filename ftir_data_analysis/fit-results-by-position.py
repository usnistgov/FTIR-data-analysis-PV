# -*- coding: utf-8 -*-
"""

accepts data csvs output by peak deconvolution script. 
input data organized as (parent directory)/3_fit-results/(normalization wavenumber)/chamber-(x)/YYYYMMDD-(z)h/YYYYMMDD_(sample material)-Ch(x)-Pos(y)-(z)h-Air_(norm wn)-fit_(replicate).csv
e.g. 3_fit-results/723/chamber-1/20250110-0h/20250110_PET-Ch1-Pos1-0h-Air_N723-fit_1.csv

input data files are csvs laid out like so: 

wavenumbers	area (N723)	FWHM	y int	height (N723)	divided Area (by630)	divided Area (by645)	...         divided Area (by1857)
631.13...	0.222..	    6.55...	5.7e-45	0.031..	        1	                    0.146...                ...	        1.39...e+37
654.09...	1.51...	    46.39.. 2.1e-46	0.03..	        6.8..	                1		                ...         9.48..e+37
...         ...         ...     ...     ...             ...                     ...                     ...         ...

@author: klj
"""


import os, itertools        #os needed for os.walk. itertools allows for listcomprehension, speficially itertools.groupby allows easy grouping of replicate files 
from pathlib import Path
import numpy as np
import shutil               #for copying files 
import re                   #regular expressions for list comprehension and string parsing 

np.set_printoptions(suppress=True)

def getPosition(file):
    return int(re.sub('[a-zA-Z]+', '', file.split('-')[2]))

def getChamber(file):
    return int(re.sub('[a-zA-Z]+', '', file.split('-')[1]))



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
            if filename[-7:-4] != "Avg" and filename.split("-")[4].split("_")[0]=="Air":
                posNum = int(filename.split("-")[2].replace("Pos", "").replace("pos", ""))
                if posNum not in posList: 
                    posList.append(posNum)
    posList.sort()
    return posList

def fitByPosition(
    rawDataDir,
    fitFTIRFolder = "3_fit-results",
    byPosFolder = "4a_fit-by-position",
    normWn = '723',
    # numMeas = 4         #to-do: get away from hard coding this 
    ):

    rawDataFolder = Path(rawDataDir)
    parentDir = rawDataFolder.absolute().parent 
    sourceFolder = parentDir / fitFTIRFolder / normWn
    byPosOutFolder = parentDir / byPosFolder / normWn

    chFolderList = makeChamberFolderList(sourceFolder)

    for chamberFolder in chFolderList:
        positionList = createPosList(chamberFolder)
        chamberID = str(chamberFolder).split("\\")[-1].split('-')[-1]

        for root, dirs, files in os.walk(chamberFolder):
            for filename in files:
                filePos = filename.split('-')[2]
                outPosFolder = byPosOutFolder / f"chamber-{chamberID}" / filePos 
                outPosFolder.mkdir(parents=True, exist_ok=True)
                shutil.copy(Path(root) / filename, outPosFolder / filename)

#creates averages and standard deviations for replicates sets and saves files. 
def createAvgStdFiles(
    rawDataDir,
    byPosFolderNm = "4a_fit-by-position",
    avgStdFolderNm = "4b_fit-averages-stdevs",
    normWn = '723',
    # numMeas = 4         #to-do: get away from hard coding this 
    ):

    rawDataFolder = Path(rawDataDir)
    parentDir = rawDataFolder.absolute().parent 
    byPosFolder = parentDir / byPosFolderNm / normWn
    avgStdFolder = parentDir / avgStdFolderNm / normWn

    for root, dirs, files in os.walk(byPosFolder):
        # depth = root.replace(str(byPosFolder), '').count(os.sep)
        if len(files)>0:
            avgStdPosFolder = Path(str(root).replace(str(byPosFolder), str(avgStdFolder))) #output folder for avg and std dev 
            fileSets = [list(g) for _, g in itertools.groupby(files, lambda x: x[:x.rfind('_')])]     #groups file replicates in list 
            
        #creates 3d arrays for each file set. 3d arrays are sets of fit results for sets of replicates.
            columnNames = ','.join(np.genfromtxt(Path(root) / fileSets[0][0], delimiter=',', names=True, max_rows=1).dtype.names) #retrieving column names. genfromtxt stores column names as dtype. join combines list into string with delimiter ','
            #local function for setArray list comprehension
            def importDataFunc(file):
                return np.loadtxt(Path(root) / file, delimiter=',', skiprows=1)
            
            for set in fileSets:  
                setArray = np.array([importDataFunc(filename) for filename in set])
                print(setArray.shape)
                filePrefix = set[0][:(set[0].rfind('_'))]

                avgArray = setArray.mean(axis=0)  #wavenumbers included in average
                stDevArray = setArray.std(axis=0, ddof=1)   #ddof is delta degrees of freedom. divisor n -ddof. using 1 since this is a sample not a population. https://numpy.org/devdocs/reference/generated/numpy.std.html 
                avgPosFolder, stdPosFolder = avgStdPosFolder / 'Average', avgStdPosFolder / 'StDev'
                avgPosFolder.mkdir(parents=True, exist_ok=True)
                stdPosFolder.mkdir(parents=True, exist_ok=True)
                np.savetxt(avgPosFolder / f"{filePrefix}_Avg.csv", avgArray, delimiter=',', header=columnNames, comments='')
                np.savetxt(stdPosFolder / f"{filePrefix}_StDev.csv", stDevArray, delimiter=',', header=columnNames, comments='')



                

if __name__ == "__main__":          #does not run if importing only if running
    fitByPosition("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-exposure/0_raw-data")    
    # createAvgStdFiles("C:/Users/klj/OneDrive - NIST/Projects/PV-Project/Reciprocity/FTIR-data-PET-exposure/0_raw-data")    
    