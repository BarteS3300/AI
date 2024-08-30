import csv
import math
import os
import matplotlib.pyplot as plt
import numpy as np  
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from mpl_toolkits import mplot3d
from myRegressor import MyLinearBinaryRegressor

def getFileData(fileName):
    crtDir = os.getcwd()
    return os.path.join(crtDir, 'data', fileName)

def loadData(fileName, inputVariabName1, inputVariabName2, outputVariabName):
    data = []
    dataNames = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                dataNames = row
            else:
                data.append(row)
            line_count += 1      
    selectedVariable1 = dataNames.index(inputVariabName1)
    selectedVariable2 = dataNames.index(inputVariabName2)
    selectedOutput = dataNames.index(outputVariabName)
    inputs1 = []
    inputs2 = []
    outputs = []
    for i in range (len(data)):
        inputs1.append(data[i][selectedVariable1])
        inputs2.append(data[i][selectedVariable2])
        outputs.append(data[i][selectedOutput])
    # inputs1 = [float(data[i][selectedVariable1]) for i in range(len(data))]
    # selectedVariable2 = dataNames.index(inputVariabName2)
    # inputs2 = [float(data[i][selectedVariable2]) for i in range(len(data))]
    # selectedOutput = dataNames.index(outputVariabName)
    # outputs = [float(data[i][selectedOutput]) for i in range(len(data))]
    
    return inputs1, inputs2, outputs

def emptyDataCorrection(data):
    mean = 0
    noOfValues = 0
    for i in range(len(data)):
        if data[i] != '':
            mean += float(data[i])
            noOfValues += 1
    mean = mean/noOfValues
    for i in range(len(data)):
        if data[i] == '':
            data[i] = mean
        data[i] = float(data[i])
    return data

def getMean(data):
    return sum(data)/len(data)

def getStandardDeviation(data):
    mean = getMean(data)
    return (1 / len(data) * sum([(p - mean)**2 for p in data]))**0.5

def normalizeData(data, mean, std):
    return [(p - mean) / std for p in data]

def plotDataHistogram(x, variableName):
    plt.hist(x, bins=10)
    plt.title('Histogram of ' + variableName)
    plt.show()
    
def plotLiniarity(x, y, xLabel, yLabel):
    plt.scatter(x, y)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(xLabel + ' vs ' + yLabel)
    plt.show()
    
def plot3D(x, y, z, xLabel, yLabel, zLabel):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_zlabel(zLabel)
    plt.title(xLabel + ' and ' + yLabel + ' vs ' + zLabel)
    plt.show()
    
def trainValidationSplit(inputs, outputs, train=0.8, validation=0.2):
    np.random.seed(5)
    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(train*len(inputs)), replace=False)
    validationSample = [i for i in indexes if i not in trainSample]
    
    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]
    
    validationInputs = [inputs[i] for i in validationSample]
    validationOutputs = [outputs[i] for i in validationSample]
    
    return trainInputs, trainOutputs, validationInputs, validationOutputs
    
def plotTrainValidation(trainInputs, trainOutputs, validationInputs, validationOutputs, xLabel, yLabel):
    plt.plot(trainInputs, trainOutputs, 'ro', label = 'Training data')
    plt.plot(validationInputs, validationOutputs, 'g^', label = 'Validation data')
    plt.title('train and validaion data')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.legend()
    plt.show()
    
def plotTrainValidation3D(trainInputs1, trainInputs2, trainOutputs, validationInputs1, validationInputs2, validationOutputs, xLabel, yLabel, zLabel):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(trainInputs1, trainInputs2, trainOutputs, color='red', label='Training data')
    ax.scatter(validationInputs1, validationInputs2, validationOutputs, color='green', label='Validation data')
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_zlabel(zLabel)
    plt.legend()
    plt.title('train and validation data')
    plt.show()
    
def trainModel(trainInputs, trainOutputs):
    regressor = linear_model.LinearRegression()
    regressor.fit(trainInputs, trainOutputs)
    return regressor

def plotModel(trainInputs, trainOutputs, w0, w1, xLabel, yLabel):
    print(len(trainInputs), len(trainOutputs))
    nOfPoints = 1000
    xref =[]
    val = min(trainInputs)
    step = (max(trainInputs) - min(trainInputs))/nOfPoints
    for i in range(1, nOfPoints):
        xref.append(val)
        val += step
    yref = [w0 + w1*x for x in xref]
    
    plt.plot(trainInputs, trainOutputs, 'ro', label = 'Training data')
    plt.plot(xref, yref, label = 'Model')
    plt.title('train data and the learnt model')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.legend()
    plt.show()
    
def plotModel3D(trainInputs, trainOutputs, w0, w1, w2, xLabel, yLabel, zLabel):
    noOfPoints = 1000
    xref = []
    xval = min([i[0] for i in trainInputs])
    step = (max([i[0] for i in trainInputs]) - min([i[0] for i in trainInputs]))/noOfPoints
    for i in range(1, noOfPoints):
        xref.append(xval)
        xval += step
    
    yref = []
    yval = min([i[1] for i in trainInputs])
    step = (max((i[1] for i in trainInputs)) - min([i[1] for i in trainInputs]))/noOfPoints
    for i in range(1, noOfPoints):
        yref.append(yval)
        yval += step
    
    zref = [w0 + w1*x + w2*y for x, y in zip(xref, yref)]
        
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter([i[0] for i in trainInputs], [i[1] for i in trainInputs], trainOutputs, color='red', label='Training data')
    ax.plot(xref, yref, zref, label='Model')
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_zlabel(zLabel)
    plt.legend()
    plt.title('train data and the learnt model')
    plt.show()
    
        
def predictInputs(inputs, regressor):
    return regressor.predict(inputs)

def plotComputedRealData(validationInputs, computedOutputs, validationOutputs, xLabel, yLabel):
    plt.plot(validationInputs, computedOutputs, 'yo', label = 'Computed data')
    plt.plot(validationInputs, validationOutputs, 'g^', label = 'Real data')
    plt.legend()
    plt.title('Computed and real data')
    plt.show()
    
def plotComputedRealData3D(validationInputs, computedOutputs, validationOutputs, xLabel, yLabel, zLabel):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter([i[0] for i in validationInputs], [i[1] for i in validationInputs], computedOutputs, color='yellow', label='Computed data')
    ax.scatter([i[0] for i in validationInputs], [i[1] for i in validationInputs], validationOutputs, color='green', label='Real data')
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_zlabel(zLabel)
    plt.legend()
    plt.title('Computed and real data')
    plt.show()

def errorPerforamnceManual(realOutputs, computedOutputs):
    error = 0
    for t1, t2 in zip(realOutputs, computedOutputs):
        error += (t1 - t2)**2
    return error/len(realOutputs)
    
def errorPerformanceSklearn(realOutputs, computedOutputs):
    return mean_squared_error(realOutputs, computedOutputs)

# def sum(mul, *arg, div):
#     sum = 0
#     for i in arg:
#         sum += i
#     return mul*sum/div
# #print(sum(2, 1, 2, 3, 4, 5, div=5))

def main():
    filePath = getFileData('v1_world-happiness-report-2017.csv')
    
    inputs1, inputs2, outputs = loadData(filePath, 'Economy..GDP.per.Capita.', 'Freedom', 'Happiness.Score')
    inputs1 = emptyDataCorrection(inputs1)
    inputs2 = emptyDataCorrection(inputs2)
    outputs = emptyDataCorrection(outputs)
    inputs = [[i, j] for i, j in zip(inputs1, inputs2)]
    
    # print('in: ', inputs)
    # print('out: ', outputs)
    maenInputs1 = getMean(inputs1)
    stdInputs1 = getStandardDeviation(inputs1)
    normalizedInputs1 = normalizeData(inputs1, maenInputs1, stdInputs1)
    
    meanInputs2 = getMean(inputs2)
    stdInputs2 = getStandardDeviation(inputs2)
    normalizedInputs2 = normalizeData(inputs2, meanInputs2, stdInputs2)
    
    meanOutputs = getMean(outputs)
    stdOutputs = getStandardDeviation(outputs)
    normalizedOutputs = normalizeData(outputs, meanOutputs, stdOutputs)
    
    normalizedInputs = [[i, j] for i, j in zip(normalizedInputs1, normalizedInputs2)]
    
    plotDataHistogram(normalizedInputs1, 'capita GDP')
    plotDataHistogram(normalizedInputs2, 'Freedom')
    plotDataHistogram(normalizedOutputs, 'Happiness score')
    
    plotLiniarity(normalizedInputs1, normalizedOutputs, 'capita GDP', 'Happiness score')
    plotLiniarity(normalizedInputs2, normalizedOutputs, 'Freedom', 'Happiness score')
    plot3D(normalizedInputs1, normalizedInputs2, normalizedOutputs, 'capita GDP', 'Freedom', 'Happiness score')
    trainInputs, trainOutputs, validationInputs, validationOutputs = trainValidationSplit(normalizedInputs, normalizedOutputs)
        
    plotTrainValidation([i[0] for i in trainInputs], trainOutputs, [i[0] for i in validationInputs], validationOutputs, 'capita GDP', 'Happiness score')
    plotTrainValidation([i[1] for i in trainInputs], trainOutputs, [i[1] for i in validationInputs], validationOutputs, 'Freedom', 'Happiness score')
    plotTrainValidation3D([i[0] for i in trainInputs], [i[1] for i in trainInputs], trainOutputs, [i[0] for i in validationInputs], [i[1] for i in validationInputs], validationOutputs, 'capita GDP', 'Freedom', 'Happiness score')
    # regressor1 = trainModel(trainInputs, trainOutputs)
    regressor = MyLinearBinaryRegressor()
    regressor.fit(trainInputs, trainOutputs)
    print('f(x1, x2) = ', regressor.intercept_, ' + ', regressor.coef_[0], ' * x1 + ', regressor.coef_[1], ' * x2')
    # print('f(x1, x2) = ', regressor1.intercept_, ' + ', regressor1.coef_[0], ' * x1 + ', regressor1.coef_[1], ' * x2')
    computedOutputs = predictInputs(validationInputs, regressor)
    plotComputedRealData([i[0] for i in validationInputs], computedOutputs, validationOutputs, 'capita GDP', 'Happiness score')
    plotComputedRealData([i[1] for i in validationInputs], computedOutputs, validationOutputs, 'Freedom', 'Happiness score')
    plotComputedRealData3D(validationInputs, computedOutputs, validationOutputs, 'capita GDP', 'Freedom', 'Happiness score')
    plotModel([i[0] for i in trainInputs], trainOutputs, regressor.intercept_, regressor.coef_[0], 'capita GDP', 'Happiness score')
    plotModel([i[1] for i in trainInputs], trainOutputs, regressor.intercept_, regressor.coef_[1], 'Freedom', 'Happiness score')
    plotModel3D(trainInputs, trainOutputs, regressor.intercept_, regressor.coef_[0], regressor.coef_[1], 'capita GDP', 'Freedom', 'Happiness score')
    
    print('prediction error (manual): ', errorPerforamnceManual(validationOutputs, computedOutputs))
    print('prediction error (sklearn): ', errorPerformanceSklearn(validationOutputs, computedOutputs))
    
main()