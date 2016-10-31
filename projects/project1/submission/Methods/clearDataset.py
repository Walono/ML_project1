# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np

def deleteNoneWantedData(tX, percentFactor) :

    """Detect and suppress column that have more than 
    'percentFactor' percent of -999 in ther
    """
    columnToSuppress = []
    columnTreshold = len(tX) * percentFactor

    for column in range(len(tX[0])) :
        columnCounter = 0
        for row in range(len(tX)) :
            if tX[row][column] == -999.0 :
                columnCounter += 1
                if columnCounter > columnTreshold :
                    columnToSuppress.append(column)
                    break

    newTX = []
    for row in tX :
        newRow = np.delete(row, columnToSuppress)
        newTX.append(newRow)

	
    return np.array(newTX)

def deleteUnwantedLine(tX) :
    "Suppress all line that contains a -999 in it"
    newTX = []
	
    #Suppress all line containing -999.0
    for row in tX :
        isWanted = True
        for columnIndex in range(len(tX[0])) :
            if row[columnIndex] == -999.0 :
                isWanted = False
                break
        if isWanted : 
            newTX.append(row)
    return np.array(newTX)	
	
def averageData(tX, tX_test=None) :
    """Calculatethe average of a column without -999 and then replace it. """
    newTX = []

    #Calculate average for each column without the -999.0
    colAverage = []
    for column in range(len(tX[0])) :
        totalAcceeptedValue = 0
        numberAcceptedValue = 0
        for row in tX :
            if row[column] != -999.0 :
                totalAcceeptedValue += row[column]
                numberAcceptedValue += 1
        if numberAcceptedValue == 0 :
            colAverage.append(0)
        else :
            colAverage.append(totalAcceeptedValue/numberAcceptedValue)
    
	#Replace each -999.0 by the coresponding column average
    for row in tX :
        for column in range(len(tX[0])) :
            if row[column] == -999.0 :
                row[column] = colAverage[column]
        newTX.append(row)
    if tX_test is not None:
        newTX_te = []
        for row in tX_test :
            for column in range(0,30) :
                if row[column] == -999.0 :
                    row[column] = colAverage[column]
            newTX_te.append(row)
        return np.array(newTX), np.array(newTX_te)    
     
    return np.array(newTX)