import load_data
import network2, network
import pickle
import os
import annotations
from os import listdir
from os.path import isfile, join
from os.path import basename
from PIL import Image,  ImageDraw
import numpy as np
from preprocess import *
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

class Trainer():

    def __init__(self, parameters={"n":0.01, "lmbda":5.0, "batch_size":10, "momentum": 0.9},
                 size=(96,160), version=0, net=None, preprocess=HOGPreprocess):

        self.parameters = parameters
        self.size= size
        self.version=version
        self.preprocess = preprocess

        if not net is None:
            self.net = net
        else:
            self.net = None

    def load_dataset(self, data_file):
        (self.trainDS, self.testDS, self.validationDS) = load_data.load_data_wrapper(data_file)

        self.file = basename(os.path.splitext(data_file)[0])

        self.create_net()

    def create_net(self):
        input_size = len(self.trainDS[0][0])
        output_size = len(self.trainDS[0][1])

        if self.net is None:
            if self.version == 0:
                self.net = network.Network((input_size,100, output_size))

            if self.version == 1:
                self.net = network2.Network((input_size,100, output_size), cost=network2.CrossEntropyCost)

            if self.version == 3:
                self.net = NeuralNet(
                    layers=[  # three layers: one hidden layer
                        ('input', layers.InputLayer),
                        ('hidden', layers.DenseLayer),
                        ('output', layers.DenseLayer),
                        ],
                    # layer parameters:
                    input_shape=(None, input_size),  # 96x96 input pixels per batch
                    hidden_num_units=500,  # number of units in hidden layer
                    output_nonlinearity=None,  # output layer uses identity function
                    output_num_units=output_size,  # 30 target values

                    # optimization method:
                    update=nesterov_momentum,
                    update_learning_rate=self.parameters["n"],
                    update_momentum=self.parameters["momentum"],

                    regression=True,  # flag to indicate we're dealing with regression problem
                    max_epochs=1000,  # we want to train this many epochs
                    verbose=1,
                    )



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

        if self.version == 3:
            X1, y1 = zip(*self.trainDS)
            X2, y2 = zip(*self.testDS)
            y2 = [load_data.vectorized_result(y) for y in y2]
            X, y = X1 + X2, y1 + tuple(y2)
            X = np.reshape(X, (len(X), len(X[0])))
            y = np.reshape(y, (len(y), len(y[0])))
            X = X.astype(np.float32)
            y = y.astype(np.float32)
            self.net.fit(X, y)


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

            src = Image.open(join(path_pos, file))
            imsize = src.size
            draw = ImageDraw.Draw(src)
            im = src.convert('L')
            print file

            for x in range(0, imsize[0]-self.size[0], offsetx) + list([imsize[0]-self.size[0]]):

                for y in range(0, imsize[1]-self.size[1], offsety) + list([imsize[1]-self.size[1]]):

                    rect  =(y, y+self.size[1], x, x+self.size[0])


                    #preprocessing
                    data = self.preprocess.processCrop(join(path_pos, file), rect)

                    data  = np.reshape(data, (len(data), 1))

                    rv = self.net.feedforward(data)
                    r =  np.argmax(rv)


                    #
                    if  r == 1 and rv[r] > 0.95:
                        print rv[r]
                        #load_data.show_image(data, size)
                        datas.append(data)
                        print (x,y,rv[r])
                        #print float(rv[r])
                        rect  =(x, y, x+self.size[0], y+self.size[1])
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

    def copy(self, trainer):
        self.net = trainer.net
        self.trainDS = trainer.trainDS
        self.testDS = trainer.testDS
        self.size = trainer.size
        self.parameters = trainer.parameters
        self.preprocess = trainer.preprocess
        self.version = trainer.version



