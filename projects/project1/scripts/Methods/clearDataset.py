# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np

def deleteNoneWantedData(tX, percentFactor) :

    #Detect all column we need to suppress depend of the percent of -999 in there
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

    #Suppress unwanted column in each row
    newTX = []
    for row in tX :
        newRow = np.delete(row, columnToSuppress)
        newTX.append(newRow)

	
    return np.array(newTX)

def deleteUnwantedLine(tX) :
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
	
def averageData(tX) :
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
    return np.array(newTX)