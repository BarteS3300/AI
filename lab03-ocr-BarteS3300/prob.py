from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from array import array
import os
from PIL import Image, ImageFilter, ImageEnhance
import sys
import time
import cv2
import os
from PIL import ImageFilter
import matplotlib.pyplot as plt



def authenticateClient():
    # Set API key.
    subscription_key = os.environ["VISION_KEY"]
    # Set endpoint.
    endpoint = os.environ["VISION_ENDPOINT"]
    # Call API
    computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))
    return computervision_client

def getROI(path):
    im =  cv2.imread(path)
    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL);
    x, y, w, h = cv2.selectROI("Select ROI", im, fromCenter=False, showCrosshair=True)
    
    print(f"{x = } {y = } {w = } {h = }")

def readTextFromImage(image_path, computervision_client):
    
    img = Image.open(image_path)

    gray_img = img.convert('L')

    blurred_img = gray_img.filter(ImageFilter.GaussianBlur(radius=1))   

    contrast_enhancer = ImageEnhance.Contrast(blurred_img)
    contrast_enhanced_img = contrast_enhancer.enhance(2.0)

    thresholded_img = contrast_enhanced_img.point(lambda p: 255 if p > 160 else 0)

    thresholded_img.save(image_path.split('.')[0] + "_preprocessed." + image_path.split('.')[1])
    
    image_path = image_path.split('.')[0] + "_preprocessed." + image_path.split('.')[1]
    
    # img = open("test1.png", "rb")
    img = open(image_path, "rb")
    read_response = computervision_client.read_in_stream(
        image=img,
        mode="Printed",
        raw=True
    )
    #print(read_response.headers)

    operation_id = read_response.headers['Operation-Location'].split('/')[-1]
    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)

    # Print the detected text, line by line
    text = []
    cords = []
    if read_result.status == OperationStatusCodes.succeeded:
        for text_result in read_result.analyze_result.read_results:
            for line in text_result.lines:
                text.append(line.text)
                vals = {}
                vals["x1"] = line.bounding_box[0]
                vals["y1"] = line.bounding_box[1]
                vals["x2"] = line.bounding_box[4]
                vals["y2"] = line.bounding_box[5]
                cords.append(vals)
                
    return (text, cords)


def verifyText(result, groundTruth, groundTruthCordinates):
    noOfCorrectLines = sum(i == j for i, j in zip(result[0], groundTruth))
    
    groundTruthCordinatesArray = []
    for cord in groundTruthCordinates:
        vals = {}
        vals["x1"] = cord[0]
        vals["y1"] = cord[1]
        vals["x2"] = cord[0] + cord[2]
        vals["y2"] = cord[1] + cord[3]
        groundTruthCordinatesArray.append(vals)
    
    for i, j in zip(result[1], groundTruthCordinatesArray):
        print("OCR bounding box: ", i, " manual bounding box: ", j)
        
    return noOfCorrectLines
        
    

def lcs(s1, s2):
    n1 = len(s1)
    n2 = len(s2)
    res = 0
    lCSubstring = [[0 for i in range(n2+1)] for j in range(n1+1)]
    for i in range(1, n1+1):
        for j in range(1, n2+1):
            if i == 0 and j == 0:
                lCSubstring[i][j] = 0
            elif s1[i-1] == s2[j-1]:
                lCSubstring[i][j] = 1 + lCSubstring[i-1][j-1]
                res = max(res, lCSubstring[i][j])
            else:
                lCSubstring[i][j] = 0
    return res

def levenshteinDistance(s1, s2, n1, n2):
    if n1 == 0:
        return n2
    if n2 == 0:
        return n1
    if s1[n1-1] == s2[n2-1]:
        return levenshteinDistance(s1, s2, n1-1, n2-1)
    return 1 + min(levenshteinDistance(s1, s2, n1-1, n2), levenshteinDistance(s1, s2, n1, n2-1), levenshteinDistance(s1, s2, n1-1, n2-1))

def hammingDistance(s1, s2):
    res = 0
    for i in range(max(len(s1), len(s2))):
            if i >= len(s1) or i >= len(s2):
                res += 1
            elif s1[i] != s2[i]: res += 1
    return res

def countWords(text):
    words = 0
    for line in text:
        words += len(line.split())
    return words

def countChars(text):
    chars = 0
    for line in text:
        chars += len(line)
    return chars

def countCharsWithOutSpaces(text):
    chars = 0
    for line in text:
        chars += len(line.replace(" ", ""))
    return chars

def werLevenshtein(detectedText, groundTruth):
    errors = 0
    for line1, line2 in zip(detectedText, groundTruth):
        errors += levenshteinDistance(line1, line2, len(line1), len(line2))
    return errors / countWords(groundTruth)

def cerLevenshtein(detectedText, groundTruth):
    errors = 0
    for line1, line2 in zip(detectedText, groundTruth):
        errors += levenshteinDistance(line1, line2, len(line1), len(line2))
    return errors / countChars(groundTruth)

def werHamming(detectedText, groundTruth):
    errors = 0
    for line1, line2 in zip(detectedText, groundTruth):
        errors += hammingDistance(line1, line2)
    return errors / countWords(groundTruth)

def cerHamming(detectedText, groundTruth):
    errors = 0
    for line1, line2 in zip(detectedText, groundTruth):
        errors += hammingDistance(line1, line2)
    return errors / countChars(groundTruth)

def printLines(result):
    for line in result:
        print(line)

def main():

    # get/define the ground truth
    groundTruth1 = ["Google Cloud", "Platform"]
    groundTruth2 = ["Succes in rezolvarea", "tEMELOR la", "LABORAtoaree de", "Inteligenta Artificiala!"]
    
    groundTruthCordinates2 = [(78, 299, 1257, 164), (131, 581, 916, 141), (82, 922, 919, 103), (106, 1129, 1348, 236)]
    
    computervision_client = authenticateClient()
    #getROI("test2.jpeg")
    detectedText1 = readTextFromImage("test1.png", computervision_client)[0]
    detectedText2 = readTextFromImage("test2.jpeg", computervision_client)[0]
    
    printLines(detectedText2)
    print(verifyText(readTextFromImage("test2.jpeg", computervision_client), groundTruth2, groundTruthCordinates2))
    
    print("WER(Levenshtein): " + str(werLevenshtein(detectedText1, groundTruth1)))
    print("CER(Levenshtein): " + str(cerLevenshtein(detectedText1, groundTruth1)))
    
    print("WER(Hamming): " + str(werHamming(detectedText1, groundTruth1)))
    print("CER(Hamming): " + str(cerHamming(detectedText1, groundTruth1)))
    
     
main()