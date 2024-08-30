from util import *

def run():
    inputs, outputs, outputsName = loadIrisData()
    # print("feature names: ", outputsName)
    # print("some input examples: ", inputs[0], inputs[50], inputs[-5])
    # print("corresponding labels: ", outputs[0], outputs[50], outputs[-5])
    
    trainInputs, trainOutputs, testInputs, testOutputs = splitData(inputs, outputs)
    dataDistributionMoreClasses(trainOutputs, outputsName)
    data2FeaturesMoreClasses(trainInputs, trainOutputs, outputsName)
    
    normalizatedTrainInputs, normalizatedTestInputs = normalization(trainInputs, testInputs)
    classifier = neural_network.MLPClassifier(hidden_layer_sizes=(5, ), activation='relu', max_iter=100, solver='sgd', verbose=10, random_state=1, learning_rate_init=.1)
    classifier.fit(normalizatedTrainInputs, trainOutputs)
    
    predictedOutputs = classifier.predict(normalizatedTestInputs)
    acc, prec, recall, cm = evalMultiClass(testOutputs, predictedOutputs, outputsName)
    
    plotConfusionMatrix(cm, outputsName, "iris classification")
    print('acc: ', acc)
    print('precision: ', prec)
    print('recall: ', recall)
    
run()