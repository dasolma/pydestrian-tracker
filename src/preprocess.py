from skimage.feature import hog
from skimage import data, color, exposure
from skimage.transform import resize
import skimage


class HOGPreprocess:

    lastfile = None
    lastIm = None
    hog = None

    @staticmethod
    def process(file):
        im = skimage.io.imread(file)
        image = color.rgb2gray(im)
        HOGPreprocess.lastfile = file
        HOGPreprocess.lastIm = im
        HOGPreprocess.hog = HOGPreprocess.getHOG(image)
        return HOGPreprocess.hog.ravel()



    @staticmethod
    def processCrop(file, rect, newsize=None):
        if file != HOGPreprocess.lastfile:
            HOGPreprocess.lastfile = file
            HOGPreprocess.lastIm = skimage.io.imread(file)
            HOGPreprocess.lastIm = color.rgb2gray(HOGPreprocess.lastIm)
            HOGPreprocess.hog = HOGPreprocess.getHOG(HOGPreprocess.lastIm)


        im = HOGPreprocess.hog
        im = im[rect[2]:rect[3],rect[0]:rect[1]]
        #image = color.rgb2gray(im)

        if newsize != None:
            #print "resizing 2"
            #im = HOGPreprocess.lastIm
            #im = color.rgb2gray(im)
            #im = im[rect[2]:rect[3],rect[0]:rect[1]]
            im   \

                = resize(im, (newsize[1], newsize[0]))
            #return HOGPreprocess.getHOG(im).ravel()


        return im.ravel()


    @staticmethod
    def getHOG(image):
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1), visualise=True)

        print "hog"
        return hog_image

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
        im = im[rect[2]:rect[3], rect[0]:rect[1]]
        image = color.rgb2gray(im)
        return image.ravel()


file = '../data/raw/INRIAPerson/Test/pos/crop001512.png'
rect = (400,800,200,800)
