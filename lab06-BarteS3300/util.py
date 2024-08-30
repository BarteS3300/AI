import warnings; warnings.simplefilter('ignore')
import csv
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
import numpy as np 

import pandas as pd
import os

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

def getFileData(filename):
    crtDir = os.getcwd()
    return os.path.join(crtDir,'SGD/data' , filename)

def loadData(filePath):
    return  pd.read_csv(filePath)

def emptyDataCorrection(data):
    for column in data.columns:
        data[column].fillna(data[column].mean(), inplace=True)

def getVariable(data, variableName):
    return data[variableName].values

def minMaxNormalization(data, min, max):
    return [(x - min) / (max - min) for x in data]

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

def oneInput(trainInputs):
    return [[el] for el in trainInputs]
    
def trainGDModel(trainInputs, trainOutputs):
    regressor = linear_model.SGDRegressor(alpha=0.01, max_iter=100)
    regressor.fit(trainInputs, trainOutputs)
    return regressor

def trainDGModelBatches(trainInputs, trainOutputs):
    # Define the number of epochs
    n_epochs = 100

    # Define the batch size
    # batch_size = 1000
    
    # Create and fit the regressor
    regressor = linear_model.SGDRegressor(learning_rate='constant', max_iter=n_epochs)

    # Loop over the data in batches
    for _ in range(n_epochs):
        regressor.partial_fit(trainInputs, trainOutputs)
    
    return regressor

def trainModel(trainInputs, trainOutputs):
    regressor = linear_model.LinearRegression()
    regressor.fit(trainInputs, trainOutputs)
    return regressor

def plotModel(trainInputs, trainOutputs, w0, w1, xLabel, yLabel):
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
    
def predictInputs(inputs, regressor):
    return regressor.predict(inputs)

def plotComputedRealData(validationInputs, computedOutputs, validationOutputs, xLabel, yLabel):
    plt.plot(validationInputs, computedOutputs, 'yo', label = 'Computed data')
    plt.plot(validationInputs, validationOutputs, 'g^', label = 'Real data')
    plt.legend()
    plt.title('Computed and real data')
    plt.show()
    

def errorPerforamnceManual(realOutputs, computedOutputs):
    error = 0
    for t1, t2 in zip(realOutputs, computedOutputs):
        error += (t1 - t2)**2
    return error/len(realOutputs)
    
def errorPerformanceSklearn(realOutputs, computedOutputs):
    return  mean_squared_error(realOutputs, computedOutputs)

def plot3D(x, y, z, xLabel, yLabel, zLabel):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_zlabel(zLabel)
    plt.title(xLabel + ' and ' + yLabel + ' vs ' + zLabel)
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
    
def plotModel3DMultiple(trainInputs, trainOutputs, w0, w1, w2, xLabel, yLabel, zLabel):
    noOfPoints = 40
    
    xref = []
    xval = min([i[0] for i in trainInputs])
    step = (max([i[0] for i in trainInputs]) - min([i[0] for i in trainInputs]))/noOfPoints
    for i in range(1, noOfPoints):
        xref.append(xval)
        xval += step
    xref.append(max([i[0] for i in trainInputs]))
        
    yref = []
    yval = min([i[1] for i in trainInputs])
    step = (max((i[1] for i in trainInputs)) - min([i[1] for i in trainInputs]))/noOfPoints
    for i in range(1, noOfPoints):
        yref.append(yval)
        yval += step
    yref.append(max([i[1] for i in trainInputs]))
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter([i[0] for i in trainInputs], [i[1] for i in trainInputs], trainOutputs, color='red', label='Training data')
    
    x_plane, y_plane = np.meshgrid(xref, yref)
    z_plane = w0 * x_plane + w1 * y_plane + w2
    
    # ax.plot_wireframe(x_plane, y_plane, z_plane, color='orange')
    ax.plot_surface(x_plane, y_plane, z_plane, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
    
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_zlabel(zLabel)
    plt.legend()
    plt.title('train data and the learnt model')
    plt.show()
    
def encodedDiagnostic(data):
    return [1 if d == 'B' else 0 for d in data]

def trainLogisticModel(trainInputs, trainOutputs):
    regressor = linear_model.LogisticRegression()
    regressor.fit(trainInputs, trainOutputs)
    return regressor

def predictMOrB(inputs, regressor):
    if(regressor.predict([inputs]) == 1):
        return 'B'
    return 'M'

def encodedFlowerClass(data):
    return [0 if d == 'Iris-setosa' else 1 if d == 'Iris-versicolor' else 2 for d in data]

def classOfFlower(inputs, regressor):
        if(regressor.predict([inputs]) == 0):
            return 'Iris-setosa'
        elif(regressor.predict([inputs]) == 1):
            return 'Iris-versicolor'
        return 'Iris-virginica'
    
def normalizeOneParam(data, min, max):
    return (data - min) / (max - min)