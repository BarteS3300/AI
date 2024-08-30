from util import *

def run():
    inputs, outputs, outputsName = loadDigitData()
    
    trainInputs, trainOutputs, testInputs, testOutputs = splitData(inputs, outputs)
    dataDistributionMoreClasses(trainOutputs, outputsName)
    print(trainInputs[0])
    trainInputsFlatten = [flatten(el) for el in trainInputs]
    testInputsFlatten = [flatten(el) for el in testInputs]
    print(trainInputsFlatten[0])
    print(len(trainInputsFlatten[0]))
    trainInputsNormalizated, testInputsNormalizated = normalization(trainInputsFlatten, testInputsFlatten)
    
    classifier = neural_network.MLPClassifier(hidden_layer_sizes=(5, ), activation='relu', max_iter=100, solver='sgd', verbose=10, random_state=1, learning_rate_init=.1)
    classifier.fit(trainInputsNormalizated, trainOutputs)
    
    computedOutputs = classifier.predict(testInputsNormalizated)
    acc, prec, recall, cm = evalMultiClass(testOutputs, computedOutputs, outputsName)
    plotConfusionMatrix(cm, outputsName, "digit classification")
    print('acc: ', acc)
    print('precision: ', prec)
    print('recall: ', recall)
    
    plotImagesAndLabels(testInputs, testOutputs, computedOutputs)
    
run()
