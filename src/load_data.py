"""

"""

#### Libraries
# Standard library
import cPickle
import gzip
import random
from random import shuffle
from PIL import Image

# Third-party libraries
import numpy as np

def load_data(file='../data/gray.p'):
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = open(file, 'rb')
    #f = open('data-process/gray.p', 'rb')
    training_data = cPickle.load(f)
    f.close()
    return (training_data)

def show_random_image(data, size):
    i = random.randint(0,len(data))
    pixels = data[i][0]
    show_image(pixels, size)

def show_image(data, size):
    pixels = data
    pixels = [int(x*255) for x in pixels]
    im = Image.new("L", size)
    im.putdata(pixels)
    im.show()

def save_image(data, name,  size):
    pixels = data
    pixels = [int(x*255) for x in pixels]
    im = Image.new("L", size)
    im.putdata(pixels)
    im.save(name)


def load_data_wrapper(file='../data/gray.p'):
    d = load_data(file)
    d = zip(d[0], d[1])

    #input length
    l = len(d[0][0])

    #Get training data 50% between pos and neg
    neg = [x for x in d if x[1] == float(0)]
    pos = [x for x in d if x[1] == float(1)]
    pos25  = pos[:int(len(pos) * 0.25)]
    neg25 = neg[:len(pos25)]

    d25 = neg25 + pos25
    shuffle(d25)

    training_inputs = [np.reshape(x, (l, 1)) for x in zip(*d25)[0]]
    training_results = [vectorized_result(x) for x in zip(*d25)[1]]
    training_data = zip(training_inputs, training_results)


    ##Rest to test and validation
    d = neg[len(neg25):] + pos[len(neg25):]
    shuffle(d)

    #j = int(len(d) * 0.5)
    j = len(d)

    test = d[:j]
    test_inputs = [np.reshape(x, (l, 1)) for x in zip(*test)[0]]
    test_results = [int(x) for x in zip(*test)[1]]
    test_data = zip(test_inputs, test_results)

    validation_data = []
    if j < len(d):
        validation = d[j:]
        validation_inputs = [np.reshape(x, (l, 1)) for x in zip(*validation)[0]]
        validation_results = [int(x) for x in zip(*validation)[1]]
        validation_data = zip(test_inputs, test_results)

    return (training_data, test_data, validation_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((2, 1))
    e[int(j)] = 1.0
    return e