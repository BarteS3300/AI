import matplotlib.pyplot as plt
from skimage import feature, io
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.filters import gaussian
from natsort import natsorted, ns
import os
from skimage.transform import resize

def loadImagesInFolder():
    crtDir = os.getcwd()
    filepath = os.path.join(crtDir, 'images')
    list_files = os.listdir(filepath)
    return list_files

def showAImagine(path):
    image = io.imread(path)
    io.imshow(image)
    io.show()
    
def resizedImagines():
    images = loadImagesInFolder()
    fig, axes = plt.subplots(len(images)//3, 3, figsize=(9, 9))
    for image in images:
        im = io.imread('images/' + image)
        image_resized = resize(im, (128, 128), anti_aliasing=True, order=0)
        axes[images.index(image) // 3, images.index(image) % 3].imshow(image_resized)
    fig.tight_layout()
    plt.show()
    
def grayScaleImages():
    images = loadImagesInFolder()
    fig, axes = plt.subplots(len(images)//3, 3, figsize=(9, 9))
    for imagine in images:
        im = io.imread('images/' + imagine)
        if len(im.shape) == 3:
            grayImage = rgb2gray(im[:,:,:3])
        if len(im.shape) == 2:
            grayImage = im
        axes[images.index(imagine) // 3, images.index(imagine) % 3].imshow(grayImage, cmap='gray')
    fig.tight_layout()
    plt.show()
        
def blurImage(path):
    image = io.imread(path)
    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    ax = axes.ravel()
    sigma = 3
    blurred = gaussian(image, sigma=(sigma, sigma), truncate=3, channel_axis=-1)
    ax[0].imshow(image)
    ax[0].set_title('Before image')
    ax[1].imshow(blurred)
    ax[1].set_title('After image')
    fig.tight_layout()
    plt.show()
    
def edgesImage(path):
    image = io.imread(path)
    if len(image.shape) == 3:
        grayImage = rgb2gray(image[:,:,:3])
    else:
        grayImage = image
    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    ax = axes.ravel()
    edgesCanny = feature.canny(grayImage, sigma=1)
    ax[0].imshow(image)
    ax[0].set_title('Before image')
    ax[1].imshow(edgesCanny)
    ax[1].set_title('After image')
    fig.tight_layout()
    plt.show()
    
        

def main():
    showAImagine('images/YOLO.jpg')
    resizedImagines()
    grayScaleImages()
    blurImage('images/YOLO.jpg')
    edgesImage('images/YOLO.jpg')
    
main()