from util import *
from MyLogicalRegressor import MyLogicalRegressor

def run():
    data = loadData(getFileData('wdbc.csv'))
    
    print("Predicting the diagnosis based on the radius1 and the texture1")
    inputVariableName1 = 'radius1'
    inputVariableName2 = 'texture1'
    outputVariableName = 'Diagnosis'
    
    input1 = getVariable(data, 'radius1')
    input2 = getVariable(data, 'texture1')
    output = getVariable(data, 'Diagnosis')
    
    meanInput1 = getMean(input1)
    stdInput1 = getStandardDeviation(input1)
    meanInput2 = getMean(input2)
    stdInput2 = getStandardDeviation(input2)

    normalizedInput1 = normalizeData(input1, meanInput1, stdInput1)
    normalizedInput2 = normalizeData(input2, meanInput2, stdInput2)
    encodedOutput = encodedDiagnostic(output)
    
    inputs = [[i, j] for i, j in zip(input1, input2)]
    normalizedInput = [[i, j] for i, j in zip(normalizedInput1, normalizedInput2)]
    
    # plotDataHistogram(input1, inputVariableName1)
    # plotDataHistogram(input2, inputVariableName2)
    # plotDataHistogram(encodedOutput, outputVariableName)
    
    # plotLiniarity(input1, encodedOutput, inputVariableName1, outputVariableName)
    # plotLiniarity(input2, encodedOutput, inputVariableName2, outputVariableName)
    
    # plot3D(input1, input2, encodedOutput, inputVariableName1, inputVariableName2, outputVariableName)
    
    trainInputs, trainOutputs, validationInputs, validationOutputs = trainValidationSplit(normalizedInput, encodedOutput)
    
    # plotTrainValidation([i[0] for i in trainInputs], trainOutputs, [i[0] for i in validationInputs], validationOutputs, inputVariableName1, outputVariableName)
    # plotTrainValidation([i[1] for i in trainInputs], trainOutputs, [i[1] for i in validationInputs], validationOutputs, inputVariableName2, outputVariableName)
    # plotTrainValidation3D([i[0] for i in trainInputs], [i[1] for i in trainInputs], trainOutputs, [i[0] for i in validationInputs], [i[1] for i in validationInputs], validationOutputs, inputVariableName1, inputVariableName2, outputVariableName)
    
    # regressor = trainLogisticModel(trainInputs, trainOutputs)
    # coefs = str(regressor.coef_[0])[1:-1].split()
    # coef_1 = float(coefs[0])
    # coef_2 = float(coefs[1])
    
    regressor = MyLogicalRegressor(learning_rate=0.000001)
    regressor.fitBinary(trainInputs, trainOutputs)
    coef_1 = regressor.coef_[0]
    coef_2 = regressor.coef_[1]
    print("f(x) = ", regressor.intercept_[0], " + ", coef_1, " * x1 + ", coef_2, " * x2")
    
    computedOutputs = predictInputs(validationInputs, regressor)
    # plotComputedRealData([i[0] for i in validationInputs], computedOutputs, validationOutputs, inputVariableName1, outputVariableName)
    # plotComputedRealData([i[1] for i in validationInputs], computedOutputs6, validationOutputs, inputVariableName2, outputVariableName)
    plotComputedRealData3D(validationInputs, computedOutputs, validationOutputs, inputVariableName1, inputVariableName2, outputVariableName)
    
    # plotModel([i[0] for i in trainInputs], trainOutputs, regressor.intercept_[0], coef_1, inputVariableName1, outputVariableName)
    # plotModel([i[1] for i in trainInputs], trainOutputs, regressor.intercept_[0], coef_2, inputVariableName2, outputVariableName)
    # plotModel3D(trainInputs, trainOutputs, regressor.intercept_[0], coef_1, coef_2, inputVariableName1, inputVariableName2, outputVariableName)
    # plotModel3DMultiple(trainInputs, trainOutputs, regressor.intercept_[0], coef_1, coef_2, inputVariableName1, inputVariableName2, outputVariableName)
    
    print("Error manual: ", errorPerforamnceManual(validationOutputs, computedOutputs))
    print("Error sklearn: ", errorPerformanceSklearn(validationOutputs, computedOutputs))
    
    # normalizedRadius = normalizeOneParam(18, meanInput1, stdInput1)
    # normalizedTexture = normalizeOneParam(10, meanInput2, stdInput2)
    # print(predictMOrB([normalizedRadius, normalizedTexture], regressor))
    print(predictMOrB([18, 10], regressor))