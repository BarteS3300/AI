from util import *
from MyNeuralNetwork import MyNeuralNetwork

def run():
    inputs = loadImagesData('train_64')
    outputs = [0 for i in range(229)] + [1 for i in range(119)]
    outputsNames = ['Normal', 'Sepia filter']
    
    inputs, outputs = shuffleData(inputs, outputs)
    
    
    trainInputs, trainOutputs, testInputs, testOutputs = splitData(inputs, outputs)
    dataDistributionMoreClasses(trainOutputs, outputsNames)
    print(len(trainOutputs))
    
    classifier = createModel()
    classifier.fit(trainInputs, trainOutputs, epochs=10, batch_size=10)
    
    computedOutputs = [1 if output > 0.5 else 0 for output in classifier.predict(testInputs)]
    acc, prec, recall, cm = evalMultiClass(testOutputs, computedOutputs, outputsNames)
    plotConfusionMatrix(cm, outputsNames, "digit classification")
    print('acc: ', acc)
    print('precision: ', prec)
    print('recall: ', recall)
    
    flattenTrainInputs = [flattenV2(img) for img in trainInputs]
    flattenTrainInputs = np.array(flattenTrainInputs)
    
    flattenTestInputs = [flattenV2(img) for img in testInputs]
    flattenTestInputs = np.array(flattenTestInputs)
    
    
    # print(flattenTrainInputs[0])
    # print(flattenTrainInputs[0].size)
    # print(trainOutputs.shape)
    
    m, n = flattenTrainInputs.shape
    flattenTrainInputs = flattenTrainInputs.T
    
    model = MyNeuralNetwork(m, n)
    model.gradient_descent(flattenTrainInputs, trainOutputs, 0.1, 100)
    
    
    flattenTestInputs = [flattenV2(img) for img in testInputs]
    flattenTestInputs = np.array(flattenTestInputs)
    flattenTestInputs = flattenTestInputs.T
    
    computedOutputs = [1 if output > 0.5 else 0 for output in model.predict(flattenTestInputs)]
    acc, prec, recall, cm = evalMultiClass(testOutputs, computedOutputs, outputsNames)
    plotConfusionMatrix(cm, outputsNames, "digit classification")
    print('acc: ', acc)
    print('precision: ', prec)
    print('recall: ', recall)
    import tensorflow as tf
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
run()