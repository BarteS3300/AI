from sklearn.preprocessing import StandardScaler
import numpy as np 
from sklearn import neural_network
import matplotlib.pyplot as plt 
from sklearn import neural_network
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
import itertools
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import cv2

def loadIrisData():
    from sklearn.datasets import load_iris
    
    data = load_iris()
    inputs = data['data']
    outputs = data['target']
    outputsName = data['target_names']
    featureNames = list(data['feature_names'])
    feature1 = [feat[featureNames.index('sepal length (cm)')] for feat in inputs]
    feature2 = [feat[featureNames.index('petal length (cm)')] for feat in inputs]
    inputs = [[feat[featureNames.index('sepal length (cm)')], feat[featureNames.index('petal length (cm)')]] for feat in inputs]
    return inputs, outputs, outputsName

def splitData(inputs, outputs):
    import numpy as np
    np.random.seed(5)
    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(0.8*len(inputs)), replace=False)
    testSample = [i for i in indexes if i not in trainSample]
    
    trinInputs = np.array([inputs[i] for i in trainSample])
    trainOutputs = np.array([outputs[i] for i in trainSample])
    testInputs = np.array([inputs[i] for i in testSample])
    testOutputs = np.array([outputs[i] for i in testSample])
    
    return trinInputs, trainOutputs, testInputs, testOutputs

def normalization(trainData, testData):
    scaler = StandardScaler()
    if not isinstance(trainData[0], list):
        trainData = [[el] for el in trainData]
        testData = [[el] for el in testData]
        
        scaler = scaler.fit(trainData)
        normalisedTrainData = scaler.transform(trainData)
        normalisedTestData = scaler.transform(testData)
        
        normalisedTrainData = [el[0] for el in trainData]
        normalisedTestData = [el[0] for el in testData]
        
    else:
        scaler = scaler.fit(trainData)
        normalisedTrainData = scaler.transform(trainData)
        normalisedTestData = scaler.transform(testData)
        
    return normalisedTrainData, normalisedTestData

def data2FeaturesMoreClasses(inputs, outputs, outputsName):
    labels = set(outputs)
    noData = len(inputs)
    for label in labels:
        x = [inputs[i][0] for i in range(noData) if outputs[i] == label]
        y = [inputs[i][1] for i in range(noData) if outputs[i] == label]
        plt.scatter(x, y, label=outputsName[label])
    plt.xlabel('sepal length (cm)')
    plt.ylabel('petal length (cm)')
    plt.legend()
    plt.show()
    
def dataDistributionMoreClasses(outputs, outputsName):
   bins = range(len(outputsName) + 1)
   plt.hist(outputs, bins, rwidth = 0.8)
   bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
   plt.xticks(np.arange(min(bins) + bin_w/2, max(bins), bin_w), outputsName)
   plt.show()

def evalMultiClass(realLabels, computedLabels, labelNames):
    confMatrix = confusion_matrix(realLabels, computedLabels)
    acc = np.sum([confMatrix[i][i] for i in range(len(labelNames))]) / len(realLabels)
    precision = {}
    recall = {}
    for i in range(len(labelNames)):
        precision[labelNames[i]] = confMatrix[i][i] / np.sum([confMatrix[j][i] for j in range(len(labelNames))])
        recall[labelNames[i]] = confMatrix[i][i] / np.sum([confMatrix[i][j] for j in range(len(labelNames))])
    
    return acc, precision, recall, confMatrix

def plotConfusionMatrix(cm, classNames, title):
    classes = classNames
    plt.figure()
    plt.imshow(cm, interpolation = 'nearest', cmap = 'Blues')
    plt.title('Confusion Matrix ' + title)
    plt.colorbar()
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)

    text_format = 'd'
    thresh = cm.max() / 2.
    for row, column in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(column, row, format(cm[row, column], text_format),
                horizontalalignment = 'center',
                color = 'white' if cm[row, column] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    plt.show()
    
def loadDigitData():
    from sklearn.datasets import load_digits
    
    data = load_digits()
    inputs = data.images
    outputs = data["target"]
    outputsName = data["target_names"]
    
    noData = len(inputs)
    permutation = np.random.permutation(noData)
    inputs = inputs[permutation]
    outputs = outputs[permutation]
    
    return inputs, outputs, outputsName

def flatten(mat):
    x = []
    for line in mat:
        for el in line:
            x.append(el)
    return x

def flattenV2(img):
    x = []
    for line in img:
        for pixel in line:
            x.append(pixel[0])
            x.append(pixel[1])
            x.append(pixel[2])
    x = np.array(x)
    return x
    
def plotImagesAndLabels(images, realLabels, computedLabels):
    n = 10
    m = 5
    fig, axes = plt.subplots(n, m, figsize=(7, 7))
    fig.tight_layout()
    for i in range(0, n):
        for j in range(0, m):
            axes[i][j].imshow(images[i * m + j])
            if(realLabels[i * m + j] == computedLabels[i * m + j]):
                color = 'green'
            else:
                color = 'red'
            axes[i][j].set_title('r: ' + str(realLabels[i * m + j]) + ' c: ' + str(computedLabels[i * m + j]), color=color)
            axes[i][j].set_axis_off()
    plt.show()

def deleteAllFromFolder(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            os.remove(os.path.join(root, file))

def applySepiaFilter(image):
    newImage = np.copy(image)
    for row in newImage:
        for pixel in range(len(row )):
            red = row[pixel][2]
            green = row[pixel][1]
            blue = row[pixel][0]
            newRed = int(red * .393 + green * .769 + blue * .189)
            newGreen = int(red * .349 + green * .686 + blue * .168)
            newBlue = int(red * .272 + green * .534 + blue * .131)
            row[pixel] = [min(newBlue, 255), min(newGreen, 255), min(newRed, 255)]
            
    return newImage

def loadImages(path):
    images = []    
    for image in os.listdir(path):
        img = cv2.imread(os.path.join(path, image))
        images.append(img)
    
    return images

def loadImagesData(path):
    import tensorflow as tf
    images = []    
    for image in os.listdir(path):
        img = tf.keras.preprocessing.image.load_img(os.path.join(path, image), target_size=(64, 64))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = img / 255
        images.append(img)
    
    return images

def transform80imagestosepia(images):
    index = [i for i in range(len(images))]
    imagesToSepia = np.random.choice(index, 80, replace=False)
    nr = 0
    for i in imagesToSepia:
        images[i] = applySepiaFilter(images[i])
        nr += 1
        print("sepia filter apply to " + str(nr) + " images")
    return images, imagesToSepia

def resizeImages(images):
    newImages = []
    for image in images:
        newImages.append(cv2.resize(image, (64, 64)))
    return newImages

def saveImages(images):
    path = 'data'
    for i in range(len(images)):
        print(str(int(i*100/len(images))) + "%")
        cv2.imwrite(os.path.join(path, 'image' + str(i) + '.jpg'), images[i])

def createModel():

    import tensorflow as tf
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(64, 64, 3)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model 

def shuffleData(inputs, outputs):
    import numpy as np
    indexes = [i for i in range(len(inputs))]
    np.random.shuffle(indexes)
    inputs = [inputs[i] for i in indexes]
    outputs = [outputs[i] for i in indexes]
    return inputs, outputs