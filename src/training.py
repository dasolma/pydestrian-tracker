from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from random import shuffle
import load_data
import network2, network
import pickle
import os
import annotations
from os import listdir
from os.path import isfile, join
from os.path import basename
from PIL import Image, ImageFilter, ImageDraw
import numpy as np

class Trainer():

    def __init__(self, parameters={"n":0.01, "lmbda":5.0, "batch_size":10}, size=(70,134), version=0, net=None):

        self.parameters = parameters
        self.size= size
        self.version=version

        if not net is None:
            self.net = net
        else:
            self.net = None

    def load_dataset(self, data_file):
        (self.trainDS, self.testDS, self.validationDS) = load_data.load_data_wrapper(data_file)
        input_size = len(self.trainDS[0][0])
        output_size = len(self.trainDS[0][1])
        self.file = basename(os.path.splitext(data_file)[0])

        if self.net is None:
            if self.version == 0:
                self.net = network.Network((input_size,100, output_size))

            if self.version == 1:
                self.net = network2.Network((input_size,100, output_size), cost=network2.CrossEntropyCost)



    def train(self, save=True, sizes=None, parameters=None ):


        if sizes == None: sizes = ( len(self.trainDS), len(self.testDS) )
        if parameters != None: self.parameters = parameters

        print "Training..."
        if self.version == 0:
             self.net.SGD(self.trainDS[:sizes[0]], 100, 10,
                           eta=self.parameters["n"],
                           test_data=self.testDS[:sizes[1]])
        if self.version == 1:
            t = self.trainDS[:sizes[0]]
            self.net.SGD(t, 100, 100,
                         eta=self.parameters["n"],
                         lmbda = self.parameters["lmbda"],
                         evaluation_data=self.testDS[:sizes[1]],
                         monitor_evaluation_accuracy=True)

        right = self.net.evaluate(self.testDS)
        error = (float(len(self.testDS))/right) - 1

        if(save):
            name = "../data/nets/%s0%02de.net" % (self.file, int(error*100))
            pickle.dump(self.net, open(name, 'wb'))




    def test_net(self, test_path='../data/raw/INRIAPerson/Test/' ):


        pa = annotations.PascalAnnotations(join(test_path, 'annotations'))

        path_pos = join(test_path,"pos")
        files = [ f for f in listdir(path_pos) if isfile(join(path_pos, f)) ]


        (offsetx, offsety) = (self.size[0]/3, self.size[1]/3)

        count = 0;
        datas = []
        for file in files:
            print file
            src = Image.open(join(path_pos, file))
            imsize = src.size
            src.thumbnail((imsize[0]/2, imsize[1]/2), Image.ANTIALIAS)
            imsize = src.size
            draw = ImageDraw.Draw(src)
            im = src.convert('L')


            for x in range(0, imsize[0]-self.size[0], offsetx) + list([imsize[0]-self.size[0]]):

                for y in range(0, imsize[1]-self.size[1], offsety) + list([imsize[1]-self.size[1]]):
                    rect  =(x,y, x+self.size[0], y+self.size[1])
                    pixels = list(im.crop(rect).getdata())

                    #preprocessing
                    imr = Image.new("L", self.size)
                    imr.putdata(pixels)
                    #imr = imr.filter(ImageFilter.FIND_EDGES)
                    #threshold = 50
                    #imr = imr.point(lambda p: p > threshold and 255)

                    #normalization [0,1]
                    data = [float(p)/255 for p in imr.getdata()]

                    #numpy array
                    data = np.reshape(data, (len(data), 1))

                    rv = self.net.feedforward(data)
                    r =  np.argmax(rv)

                    #
                    if  r == 1:
                        #load_data.show_image(data, size)
                        datas.append(data)
                        print (x,y,rv, r)
                        #print float(rv[r])
                        draw.rectangle(rect, outline=(255,0,0))




            del draw

            src.show(title=file)

            count += 1
            if count == 10:
                break



        print len(datas)
        return datas



    def neg(self):
        return [x for x in self.testDS if x[1] == float(0)]

    def pos(self):
        return [x for x in self.testDS if x[1] == float(1)]

    def evalutate(self):
        print "Positives: %d / %d"%(self.net.evaluate(self.pos()), len(self.pos()))
        print "Negatives: %d / %d"%(self.net.evaluate(self.neg()), len(self.neg()))



