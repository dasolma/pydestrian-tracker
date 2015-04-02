from PIL import Image
import pickle
import random
from os import listdir
from os.path import isfile, join
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure
import skimage
import threading
import time
import numpy
import sys


class HOGPreprocess:

    lastfile = None
    lastIm = None

    @staticmethod
    def process(file):
        im = skimage.io.imread(file)
        image = color.rgb2gray(im)
        lastfile = file
        lastIm = im
        return HOGPreprocess.getHOG(image)



    @staticmethod
    def processCrop(file, rect):
        if file != HOGPreprocess.lastfile:
            HOGPreprocess.lastIm = skimage.io.imread(file)
            HOGPreprocess.lastfile = file

        im = HOGPreprocess.lastIm
        im = im[rect[2]:rect[3],rect[0]:rect[1]]
        image = color.rgb2gray(im)
        return HOGPreprocess.getHOG(image)


    @staticmethod
    def getHOG(image):
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1), visualise=True)

        return hog_image.ravel()

        '''
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

        ax1.axis('off')
        ax1.imshow(image, cmap=plt.cm.gray)
        ax1.set_title('Input image')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()
        '''

class GrayPreprocess:

    lastfile = None
    lastIm = None

    @staticmethod
    def process(file):
        im = skimage.io.imread(file)
        image = color.rgb2gray(im)
        return image.ravel()



    @staticmethod
    def processCrop(file, rect):
        if file != HOGPreprocess.lastfile:
            GrayPreprocess.lastIm = skimage.io.imread(file)
            GrayPreprocess.lastfile = file

        im = GrayPreprocess.lastIm
        im = im[rect[0]:rect[1], rect[2]:rect[3]]
        image = color.rgb2gray(im)
        return image.ravel()
