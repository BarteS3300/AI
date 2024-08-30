from util import *
from MyGDRegressor import MyGDRegressor

def run():
    
    data = loadData(getFileData('world-happiness-report-2017.csv'))
    
    print("Predicting the happiness score based on the GDP per capita")
    inputVariableName1 = 'Economy..GDP.per.Capita.'
    outputVariableName = 'Happiness.Score'
    input = getVariable(data, 'Economy..GDP.per.Capita.')
    output = getVariable(data, 'Happiness.Score')
    
    minInput = min(input)
    maxInput = max(input)
    minOutput = min(output)
    maxOutput = max(output)
    
    meanInput = getMean(input)
    stdInput = getStandardDeviation(input)
    meanOutput = getMean(output)
    stdOutput = getStandardDeviation(output)
    
    normalizatedInput = normalizeData(input, meanInput, stdInput)
    normalizatedOutput = normalizeData(output, meanOutput, stdOutput)
    
    plotDataHistogram(input, inputVariableName1)
    # plotDataHistogram(normalizatedInput, inputVariableName1)
    plotDataHistogram(output, outputVariableName)
    
    plotLiniarity(normalizatedInput, normalizatedOutput, inputVariableName1, outputVariableName)
    trainInputs, trainOutputs, validationInputs, validationOutputs = trainValidationSplit(normalizatedInput, normalizatedOutput)
    # trainInputs, trainOutputs, validationInputs, validationOutputs = minMaxNormalization(trainInputs, minInput, maxInput), minMaxNormalization(trainOutputs, minOutput, maxOutput), minMaxNormalization(validationInputs, minInput, maxInput), minMaxNormalization(validationOutputs, minOutput, maxOutput)
    
    plotTrainValidation(trainInputs, trainOutputs, validationInputs, validationOutputs, inputVariableName1, outputVariableName)
    
    plotDataHistogram(input, inputVariableName1)
    
    # regressor = trainDGModelBatches(oneInput(trainInputs), trainOutputs)
    regressor = MyGDRegressor()
    regressor.fitBatch1Param(oneInput(trainInputs), trainOutputs)
    print("f(x) = ", regressor.intercept_[0], " + ", regressor.coef_[0], " * x")
    
    plotModel(trainInputs, trainOutputs, regressor.intercept_[0], regressor.coef_[0], inputVariableName1, outputVariableName)
    computedOutputs = predictInputs(oneInput(validationInputs), regressor)
    plotComputedRealData(validationInputs, computedOutputs, validationOutputs, inputVariableName1, outputVariableName)
    
    print("Error manual: ", errorPerforamnceManual(validationOutputs, computedOutputs))
    print("Error sklearn: ", errorPerformanceSklearn(validationOutputs, computedOutputs))
    
    print(".......................................................................................")
    
    print("Predicting the happiness score based on the GDP per capita and the Freedom")
    inputVariableName1 = 'Economy..GDP.per.Capita.'
    inputVariableName2 = 'Freedom'
    outputVariableName = 'Happiness.Score'
    input1 = getVariable(data, 'Economy..GDP.per.Capita.')
    input2 = getVariable(data, 'Freedom')
    output = getVariable(data, 'Happiness.Score')

    meanInput1 = getMean(input1)
    stdInput1 = getStandardDeviation(input1)
    meanInput2 = getMean(input2)
    stdInput2 = getStandardDeviation(input2)
    meanOutput = getMean(output)
    stdOutput = getStandardDeviation(output)
    
    normalizedInput1 = normalizeData(input1, meanInput1, stdInput1)
    normalizedInput2 = normalizeData(input2, meanInput2, stdInput2)
    normalizedOutput = normalizeData(output, meanOutput, stdOutput)
    inputs = [[i, j] for i, j in zip(input1, input2)]
    noramlizedInputs = [[i, j] for i, j in zip(normalizedInput1, normalizedInput2)]
    
    plotDataHistogram(input1, inputVariableName1)
    plotDataHistogram(input2, inputVariableName2)
    plotDataHistogram(output, outputVariableName)
    
    plotLiniarity(input1, output, inputVariableName1, outputVariableName)
    plotLiniarity(input2, output, inputVariableName2, outputVariableName)
    
    plot3D(input1, input2, output, inputVariableName1, inputVariableName2, outputVariableName)
    
    trainInputs, trainOutputs, validationInputs, validationOutputs = trainValidationSplit(noramlizedInputs, normalizedOutput)
    
    
    
    plotTrainValidation([i[0] for i in trainInputs], trainOutputs, [i[0] for i in validationInputs], validationOutputs, inputVariableName1, outputVariableName)
    plotTrainValidation([i[1] for i in trainInputs], trainOutputs, [i[1] for i in validationInputs], validationOutputs, inputVariableName2, outputVariableName)
    plotTrainValidation3D([i[0] for i in trainInputs], [i[1] for i in trainInputs], trainOutputs, [i[0] for i in validationInputs], [i[1] for i in validationInputs], validationOutputs, inputVariableName1, inputVariableName2, outputVariableName)
    
    #regressor = trainDGModelBatches(trainInputs, trainOutputs)
    regressor = MyGDRegressor(learning_rate=0.0001)
    regressor.fitBatch2Params(trainInputs, trainOutputs,)
    print('f(x) = ', regressor.intercept_[0], ' + ', regressor.coef_[0], ' * x1 + ', regressor.coef_[1], ' * x2')
    
    computedOutputs = predictInputs(validationInputs, regressor)
    plotComputedRealData([i[0] for i in validationInputs], computedOutputs, validationOutputs, inputVariableName1, outputVariableName)
    plotComputedRealData([i[1] for i in validationInputs], computedOutputs, validationOutputs, inputVariableName2, outputVariableName)
    plotComputedRealData3D(validationInputs, computedOutputs, validationOutputs, inputVariableName1, inputVariableName2, outputVariableName)
    
    plotModel([i[0] for i in trainInputs], trainOutputs, regressor.intercept_[0], regressor.coef_[0], inputVariableName1, outputVariableName)
    plotModel([i[1] for i in trainInputs], trainOutputs, regressor.intercept_[0], regressor.coef_[1], inputVariableName2, outputVariableName)
    plotModel3D(trainInputs, trainOutputs, regressor.intercept_[0], regressor.coef_[0], regressor.coef_[1], inputVariableName1, inputVariableName2, outputVariableName)
    plotModel3DMultiple(trainInputs, trainOutputs, regressor.intercept_[0], regressor.coef_[0], regressor.coef_[1], inputVariableName1, inputVariableName2, outputVariableName)
    
    print('Error manual: ', errorPerforamnceManual(validationOutputs, computedOutputs))
    print('Error sklearn: ', errorPerformanceSklearn(validationOutputs, computedOutputs))
    