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


nthreads = 16
nwindows = 10
input = []
output = []
size = (160,96)
dir_images = "../../data/raw/INRIAPerson/train_64x128_H96/pos"

def getHOGfromfile(file):
    im = skimage.io.imread(file)
    image = color.rgb2gray(im)
    return getHOG(image)

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

def create(files, output_value, thread, windows = 0):

    count = 1
    for file in files:
        if windows == 0:
            pixels = getHOGfromfile(join(dir_images, file))

            input.append(pixels)
            output.append(output_value)


        else:
            im = skimage.io.imread(join(dir_images, file))
            im = color.rgb2gray(im)
            for i in xrange(0,windows):

                width, height = im.shape
                x = random.randint(0, width-size[0])
                y = random.randint(0, height-size[1])

                pixels = getHOG( im[x:(x+size[0]), y:(y+size[1]) ] )

                input.append(pixels)
                output.append(output_value)


        #im = Image.open(join(dir_images,file))
        #if size is None:
        #    size = im.size


        #im = im.convert("L")

        #im = im.filter(ImageFilter.CONTOUR).filter(ImageFilter.SMOOTH).convert("L")
        #threshold = 50
        #im = im.point(lambda p: p > threshold and 255)

        #pixels = list(im.getdata())
        #pixels = [float(x)/255 for x in pixels]
        #width, height = im.size

        #pixels = [float(x)/255 for x in pixels]pixels = [pixels[i * width:(i + 1) * width] for i in xrange(height)]
        #pixels = pixels[0][0]
        #pixels = [pixels[i][0] for i in xrange(height*width)]

        print("Thread %d: %d / %d"%(thread, count, len(files)))

        count += 1

def process(files, output_value, windows=0):
    thread = 1
    chunk_size = len(files) / nthreads
    print "Chunk zise %d"%chunk_size
    for l in chunks(files, chunk_size):
        print "Creating thread %d"%(thread)
        t = threading.Thread(target=create, args = (l, output_value, thread, windows))
        t.daemon = True
        t.start()
        thread += 1

def chunks(l, n):
    n = max(1, n)
    return [l[i:i + n] for i in range(0, len(l), n)]


def go():
    global dir_images
    ### POSITIVE SAMPLES
    print "Creating positive samples...."
    #dir_images = sys.argv[1]
    dir_images = "../../data/raw/INRIAPerson/train_64x128_H96/pos"
    files = [ f for f in listdir(dir_images) if isfile(join(dir_images,f)) ]
    process(files, (float(1)))


    while( len(input) < len(files)):
        time.sleep(1)

    pickle.dump((input, output), open('hog_pos.p', 'wb'))

    ### NEGATIVE SAMPLES
    print "Creating negative samples...."
    #dir_images = sys.argv[2]
    dir_images = '../../data/raw/INRIAPerson/train_64x128_H96/neg'
    files = [ f for f in listdir(dir_images) if isfile(join(dir_images,f)) ]
    process(files, (float(0)), windows=nwindows)

    while( len(input) < len(files)):
        time.sleep(1)

    pickle.dump((input, output), open('hog.p', 'wb'))

    '''
    #get positive samples

    files = [ f for f in listdir(dir_images) if isfile(join(dir_images,f)) ]



    size = None
    for file in files:

        pixels = getHOG(join(dir_images, file))
        #im = Image.open(join(dir_images,file))
        #if size is None:
        #    size = im.size


        #im = im.convert("L")

        #im = im.filter(ImageFilter.CONTOUR).filter(ImageFilter.SMOOTH).convert("L")
        #threshold = 50
        #im = im.point(lambda p: p > threshold and 255)

        #pixels = list(im.getdata())
        #pixels = [float(x)/255 for x in pixels]
        #width, height = im.size

        #pixels = [float(x)/255 for x in pixels]pixels = [pixels[i * width:(i + 1) * width] for i in xrange(height)]
        #pixels = pixels[0][0]
        #pixels = [pixels[i][0] for i in xrange(height*width)]

        input.append(pixels)
        output.append((float(1)))



    #get negative samples
    dir_images = sys.argv[2]
    files = [ f for f in listdir(dir_images) if isfile(join(dir_images,f)) ]
    for file in files:
        print file
        for i in xrange(0,10):

            pixels = getHOG(join(dir_images, file))

            # im = Image.open(join(dir_images,file))
            #
            # im = im.convert("L")
            # #im = im.filter(ImageFilter.CONTOUR).filter(ImageFilter.SMOOTH).convert("L")
            # #threshold = 50
            # #im = im.point(lambda p: p > threshold and 255)
            #
            # width, height = im.size
            # x = random.randint(0, width-size[0])
            # y = random.randint(0, height-size[1])
            #
            # pixels = list(im.crop((x, y, x+size[0], y+size[1])).getdata())
            # pixels = [float(x)/255 for x in pixels]

            #print len(pixels)
            input.append(pixels)
            output.append((float(0)))

        print file


    pickle.dump((input, output), open('hog.p', 'wb'))

    '''
