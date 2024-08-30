from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes

import cv2
from msrest.authentication import CognitiveServicesCredentials
from array import array
import os
from PIL import Image
import sys
import time
import matplotlib.pyplot as plt
from skimage import io
import numpy as np

def authenticate_client():
    '''
    Authenticate
    Authenticates your credentials and creates a client.
    '''
    subscription_key = os.environ["VISION_KEY"]
    endpoint = os.environ["VISION_ENDPOINT"]
    computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))
    '''
    END - Authenticate
    '''
    return computervision_client

def process_image(image_path):
    im = plt.imread(image_path)
    fig, ax = plt.subplots()
    im = ax.imshow(im)
    plt.show()
    
def getROI(path):
    '''
    ROI = region of interestS
    '''
    im =  cv2.imread(path)
    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL);
    x, y, w, h = cv2.selectROI("Select ROI", im, fromCenter=False, showCrosshair=True)
    
    return x, y, w, h
    
def get_result(computervision_client, image_path):
    img = open(image_path, "rb")
    result = computervision_client.analyze_image_in_stream(img, visual_features=[VisualFeatureTypes.tags, VisualFeatureTypes.objects])
    # print("tags ")
    # for tag in result.tags:
    #     print(tag)
    #     if (tag.name == "bike") or (tag.name == "person") or (tag.name == "human") or (tag.name == "man"):
    #         print("People detected: ", tag.confidence)
    # print("objects ")
    # for ob in result.objects:
    #     print(ob.object_property, ob.rectangle)
    return result
    
    
def load_images_from_folder(image_folder_path):
    crtDir = os.getcwd()
    filepath = os.path.join(crtDir, image_folder_path)
    list_files = os.listdir(filepath)
    return list_files

def get_results(computervision_client, directory):
    images = load_images_from_folder(directory)
    results = {}
    for image in images:
        result = get_result(computervision_client, directory + "/" + image)
        results[image] = result
    return results

def verify_bike(results):
    computed_bikes = {}
    for image in results.keys():
        bike_detected = False
        for tag in results[image].tags:
            if(tag.name == "bike" or tag.name == "bicycle"):
                bike_detected = True
                break
        computed_bikes[image] = bike_detected
    return computed_bikes

def eval_classification(real_labels, computed_labels, pos, neg):
    acc = sum([1 if real_labels[i] == computed_labels[i] else 0 for i in real_labels.keys()]) / len(real_labels)
    
    TP = sum([1 if (real_labels[i] == pos and computed_labels[i] == pos) else 0 for i in real_labels.keys()])
    FP = sum([1 if (real_labels[i] == neg and computed_labels[i] == pos) else 0 for i in real_labels.keys()])
    TN = sum([1 if (real_labels[i] == neg and computed_labels[i] == neg) else 0 for i in real_labels.keys()])
    FN = sum([1 if (real_labels[i] == pos and computed_labels[i] == neg) else 0 for i in real_labels.keys()])
    
    precPoz = TP / (TP + FP)
    precNeg = TN / (TN + FN)
    
    recallPoz = TP / (TP + FN)
    recallNeg = TN / (TN + FP)
    
    return acc, [precPoz, precNeg] , [recallPoz, recallNeg]

def get_coordinates_of_bikes(results):
    computed_bb = {}
    for image in results.keys():
        for obj in results[image].objects:
            arrays_bb = []
            if(obj.object_property == "bike" or obj.object_property == "bicycle" or obj.object_property == "cycle"):
                arrays_bb.append([obj.rectangle.x, obj.rectangle.y, obj.rectangle.w, obj.rectangle.h])
        if len(arrays_bb) != 0:
            computed_bb[image] = arrays_bb
    return computed_bb

def draw_bb(directory, real_bb, computed_bb):
    fig, axes = plt.subplots(int(len(real_bb) / 5 if len(real_bb) % 5 == 0 else len(real_bb) / 5 + 1), 5, figsize=(9, 9))
    i = 0
    j = 0
    for image in real_bb.keys():    
        im = io.imread(directory + "/" + image)
        axes[i, j].imshow(im)
        for bb in real_bb[image]:
            x, y, w, h = bb
            axes[i, j].add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', lw=2))
        if image in computed_bb.keys():
            for bb in computed_bb[image]:
                x, y, w, h = bb
                axes[i, j].add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='blue', lw=2))
        j = j + 1
        if j == 5:
            i = i + 1
            j = 0
    fig.tight_layout()
    plt.show()
    
def manual_ROI(directory):
    images = load_images_from_folder(directory)
    real_bb = {}
    for image in images:
        bb = []
        comand = "A"
        x, y, w, h = 0, 0, 0, 0
        while(comand != 'N' and comand != 'E'):
            if(comand == 'S'):
                bb.append([x, y, w, h])
            x, y, w, h = getROI(directory + "/" + image) 
            print(f"{x = } {y = } {w = } {h = }")
            comand = input("Save: S, Again: A, Next: N, Exit: E\n")
        if(comand == 'E'):
            return real_bb
        if(len(bb) != 0):
            real_bb[image] = bb
    return real_bb

def error_bbv1(real_bb, computed_bb):
    IoU = 0
    total_real_bb = 0
    total_computed_bb = 0
    for image in real_bb.keys():
        total_real_bb += len(real_bb[image])
        if image in computed_bb:
            i = 0
            for bb in real_bb[image]:
                if i < len(computed_bb[image]):
                    total_computed_bb += 1
                    rx, ry, rw, rh = bb
                    cx, cy, cw, ch = computed_bb[image][i]
                    intersection_w = max(rx, cx) - min(rx + rw, cx + cw)
                    intersection_h = max(ry, cy) - min(ry + rh, cy + ch)
                    aria = intersection_w * intersection_h
                    IoU = IoU + aria / (rw * rh + cw * ch - aria)
                    i += 1
                    
    return IoU/total_real_bb

def error_bbv2(real_bb, computed_bb):
    error = 0
    total_real_bb = 0
    total_computed_bb = 0
    for image in real_bb.keys():
        total_real_bb += len(real_bb[image])
        if image in computed_bb:
            i = 0
            for rbb in real_bb[image]:
                if i < len(computed_bb[image]):
                    total_computed_bb += 1
                    cbb = computed_bb[image][i]
                    for v in zip(rbb, cbb):
                        error = error + abs(v[0] - v[1])**2
                    error /= 4
                    i += 1
        #         else:
        #             error += rbb[0] + rbb[1] + rbb[2] + rbb[3]
        #             error /= 4
        # else:
        #     for rbb in real_bb[image]:
        #         error += rbb[0] + rbb[1] + rbb[2] + rbb[3]
        #         error /= 4
                
    return error/total_real_bb
                
def main():
    directory = "bikes"
    real_labels = {'bike02.jpg': True, 'bike03.jpg': True, 'bike04.jpg': True, 'bike05.jpg': True, 'bike06.jpg': True, 'bike07.jpg': True, 'bike08.jpg': True, 'bike09.jpg': True, 'bike1.jpg': True, 'bike10.jpg': True, 'traffic01.jpg': False, 'traffic02.jpg': False, 'traffic03.jpg': False, 'traffic04.jpg': False, 'traffic05.jpg': False, 'traffic06.jpg': False, 'traffic07.jpg': False, 'traffic08.jpg': False, 'traffic09.jpg': False, 'traffic10.jpg': False}
    real_bb = {'bike02.jpg': [[16, 88, 365, 236]], 'bike03.jpg': [[157, 145, 189, 265], [63, 142, 134, 251]], 'bike04.jpg': [[0, 0, 416, 416]], 'bike05.jpg': [[69, 49, 288, 299]], 'bike06.jpg': [[157, 145, 188, 262], [65, 141, 132, 253]], 'bike07.jpg': [[59, 205, 240, 211]], 'bike08.jpg': [[53, 0, 335, 357]], 'bike09.jpg': [[4, 7, 377, 403]], 'bike1.jpg': [[5, 33, 405, 374]], 'bike10.jpg': [[141, 125, 235, 283]]}
    
    computervision_client = authenticate_client()
    
    # getTags(computervision_client, "bikes/traffic06.jpg")
    # process_image("bikes/traffic06.jpg")
    results = get_results(computervision_client, directory)
    computed_result = verify_bike(results)
    print("Detected bikes with computer vision: ", computed_result)
    acc, prec, recall = eval_classification(real_labels, computed_result, True, False)
    print("Accuracy: ", acc, " Precision: ", prec, " Recall: ", recall)
    
    computed_bb = get_coordinates_of_bikes(results)
    print(computed_bb)
    
    draw_bb(directory, real_bb, computed_bb)
    
    print("Avarage IoU: ", error_bbv1(real_bb, computed_bb))
    print("Avarage error: ", error_bbv2(real_bb, computed_bb))
    
    # print(manual_ROI(directory))
    
    
    
main()