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
from preprocess import *
import numpy
import sys


nthreads = 4
nwindows = 10
input = []
output = []
size = (160,96)
dir_images = "../../data/raw/INRIAPerson/train_64x128_H96/pos"

def create(files, output_value, thread, windows = 0, preprocess=GrayPreprocess):

    count = 1
    for file in files:
        #print "File %s"%(join(dir_images, file))
        if windows == 0:
            pixels = preprocess.process(join(dir_images, file))


            input.append(pixels)
            output.append(output_value)


        else:
            im = skimage.io.imread(join(dir_images, file))
            im = color.rgb2gray(im)
            for i in xrange(0,windows):

                width, height = im.shape
                x = random.randint(0, width-size[0])
                y = random.randint(0, height-size[1])

                pixels = preprocess.processCrop(join(dir_images, file), (x, x+size[0], y, y+size[1]))

                input.append(pixels)
                output.append(output_value)



        print("Thread %d: %d / %d"%(thread, count, len(files)))

        count += 1


def process(files, output_value, windows=0, preprocess=GrayPreprocess):
    thread = 1
    chunk_size = len(files) / nthreads
    print "Chunk size %d"%chunk_size
    for l in chunks(files, chunk_size):
        print "Creating thread %d"%(thread)
        t = threading.Thread(target=create, args = (l, output_value, thread, windows, preprocess))
        t.daemon = True
        t.start()
        thread += 1

def chunks(l, n):
    n = max(1, n)
    return [l[i:i + n] for i in range(0, len(l), n)]


def go(name='data.p', preprocess=GrayPreprocess, test=False):
    global dir_images
    ### POSITIVE SAMPLES
    print "Creating positive samples...."
    #dir_images = sys.argv[1]
    dir_images = "../data/raw/INRIAPerson/train_64x128_H96/pos"
    files = [ f for f in listdir(dir_images)  ] #if isfile(join(dir_images,f))
    if test == True:
        files = [files[0]]

    print len(files)
    process(files, (float(1)), preprocess=preprocess)


    while( len(input) < len(files)):
        time.sleep(1)

    ### NEGATIVE SAMPLES
    print "Creating negative samples...."
    #dir_images = sys.argv[2]
    dir_images = '../data/raw/INRIAPerson/train_64x128_H96/neg'
    files = [ f for f in listdir(dir_images) ] #if isfile(join(dir_images,f))
    if test == True:
        files = [files[0]]
    process(files, (float(0)), windows=nwindows, preprocess=preprocess)


    while( len(output) < (len(files) * nwindows)):
        print "%d < %d"%(len(output), (len(files) * nwindows))
        time.sleep(1)

    print "Saving.... %s"%(name)
    pickle.dump((input, output), open(name, 'wb'))

    return (input, output)


