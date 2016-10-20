# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np

#####TODO: Suppress this part
from Methods.proj1_helpers import *
DATA_TRAIN_PATH = 'csv/train.csv' # TODO: download train data and supply path here 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
tX_tra = tX.T
COLUMN_PERCENT = 0.5
#####


def deleteNoneWantedData(tX, factor) :

    #Detect all column we need to suppress depend of the percent of -999 in there
    columnToSuppress = []
    columnTreshold = len(tX) * factor

    for column in range(len(tX[0])) :
        columnCounter = 0
        for row in range(len(tX)) :
            if tX[row][column] == -999.0 :
                columnCounter += 1
                if columnCounter > columnTreshold :
                    columnToSuppress.append(column)
                    break

    #Suppress unwanted column in each row
    newTX = []
    for row in tX :
        newRow = np.delete(row, columnToSuppress)
        newTX.append(newRow)

	
    return newTX

def deleteUnwantedLine(tX) :
    newTX = []
	
    #Suppress all line containing -999
    for row in tX :
        isWanted = True
        for columnIndex in range(len(tX[0])) :
            if row[columnIndex] == -999.0 :
                isWanted = False
                break
        if isWanted : 
            newTX.append(row)
    return newTX	
	
#####TODO: Suppress this part
testT = deleteNoneWantedData(tX, COLUMN_PERCENT)
print(str(len(testT[0])))
testT2 = deleteUnwantedLine(testT)
print(str(len(testT2)))
testT3 = deleteUnwantedLine(tX)
print(str(len(testT3)))
#####