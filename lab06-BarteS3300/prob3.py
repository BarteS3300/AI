from util import *
from MyLogicalRegressor import MyLogicalRegressor

def run():
    data = loadData(getFileData('iris.csv'))
    
    print("Predicting the species of a flower")
    inputVariableName1 = 'sepal_lenght'
    inputVariableName2 = 'sepal_width'
    inputVariableName3 = 'petal_length'
    inputVariableName4 = 'petal_width'
    outputVariableName = 'class'
    
    input1 = getVariable(data, inputVariableName1)
    input2 = getVariable(data, inputVariableName2)
    input3 = getVariable(data, inputVariableName3)
    input4 = getVariable(data, inputVariableName4)
    output = getVariable(data, outputVariableName)
    
    meanInput1 = getMean(input1)
    stdInput1 = getStandardDeviation(input1)
    meanInput2 = getMean(input2)
    stdInput2 = getStandardDeviation(input2)
    meanInput3 = getMean(input3)
    stdInput3 = getStandardDeviation(input3)
    meanInput4 = getMean(input4)
    stdInput4 = getStandardDeviation(input4)

    normalizedInput1 = normalizeData(input1, meanInput1, stdInput1)
    normalizedInput2 = normalizeData(input2, meanInput2, stdInput2)
    normalizedInput3 = normalizeData(input3, meanInput3, stdInput3)
    normalizedInput4 = normalizeData(input4, meanInput4, stdInput4)
    encodedOutput = encodedFlowerClass(output)
    
    inputs = [[i, j, n, m] for i, j, n, m in zip(input1, input2, input3, input4)]
    normalizedInput = [[i, j, n, m] for i, j, n, m in zip(normalizedInput1, normalizedInput2, normalizedInput3, normalizedInput4)]
    
    trainInputs, trainOutputs, validationInputs, validationOutputs = trainValidationSplit(normalizedInput, encodedOutput)
    
    regressor = trainLogisticModel(trainInputs, trainOutputs)
    print(regressor.coef_)
    # coefs = str(regressor.coef_[0])[1:-1].split()
    # coef_1 = float(coefs[0])
    # coef_2 = float(coefs[1])
    # coef_3 = float(coefs[2])
    # coef_4 = float(coefs[3])
    
    regressor = MyLogicalRegressor(learning_rate=0.000001)
    regressor.fit(trainInputs, trainOutputs)
    # coef_1 = regressor.coef_[0]
    # coef_2 = regressor.coef_[1]
    # coef_3 = regressor.coef_[2]
    # coef_4 = regressor.coef_[3]
    
    # print("f(x) = ", regressor.intercept_[0], " + ", coef_1, " * x1 + ", coef_2, " * x2", " + ", coef_3, " * x3", " + ", coef_4, " * x4")
    
    computedOutputs = predictInputs(validationInputs, regressor)
    
    print("Error manual: ", errorPerforamnceManual(validationOutputs, computedOutputs))
    print("Error sklearn: ", errorPerformanceSklearn(validationOutputs, computedOutputs))
    
    # sepalLengthNormalized = normalizeOneParam(5.35, meanInput1, stdInput1)
    # sepalWidthNormalized = normalizeOneParam(3.85, meanInput2, stdInput2)
    # petalLengthNormalized = normalizeOneParam(1.25, meanInput3, stdInput3)
    # petalWidthNormalized = normalizeOneParam(0.4, meanInput4, stdInput4)
    # print(classOfFlower([sepalLengthNormalized, sepalWidthNormalized, petalLengthNormalized, petalWidthNormalized], regressor))
    print(classOfFlower([5.35, 3.85, 1.25, 0.4], regressor))