#Constant
COLUMN_PERCENT = 0.5

#Create a table of all line ine train.csv without the \n
with open('train.csv') as f:
    trainLine = [line.rstrip('\n') for line in f]

#Split all line in a list of list
trainSplitedLine = []
for elem in trainLine : trainSplitedLine.append(elem.split(","))

#Add in a list all column we need to supress
columnToSuppress = []
columnTreshold = len(trainLine) * COLUMN_PERCENT
for i in range(2, len(trainSplitedLine[0])) :
    numberOfNull = 0
    for elem in trainSplitedLine[i] :
        if elem == "-999" :
            numberOfNull = numberOfNull + 1
            if numberOfNull < columnTreshold :
                columnToSuppress.append(i)
                break

#suppress all unwanted column
sizeOfSuppression = len(columnToSuppress) - 1
for i in range(sizeOfSuppression) :
    for row in trainSplitedLine :
        del row[columnToSuppress[sizeOfSuppression - i]]

#Create and write in sortedFile.csv all column that don't contains "-999"
sortedFile = open('sortedFile.csv', 'w')
for row in range(len(trainSplitedLine)) :
    currentRow = ""
    for column in range(len(trainSplitedLine[row])) :
        if trainSplitedLine[row][column] == "-999" :
            currentRow = ""
            break
        if column == len(trainSplitedLine[row]) - 1 :
            currentRow = currentRow + trainSplitedLine[row][column] + '\n'
        else :
            currentRow = currentRow + trainSplitedLine[row][column] + ","
    sortedFile.write(currentRow)
sortedFile.close()