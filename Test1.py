import numpy as np
import Layers
import random
import time
import idx2numpy

import gzip
import random
import time
training_data = gzip.open(r'C:\Users\linus\train-images-idx3-ubyte.gz', 'r')        #första 'r' gör att den inte letar efter t.ex \n
training_answers = gzip.open(r'C:\Users\linus\train-labels-idx1-ubyte.gz', 'r')
testing_data = gzip.open(r'C:\Users\linus\t10k-images-idx3-ubyte.gz', 'r')
testing_answers = gzip.open(r'C:\Users\linus\t10k-labels-idx1-ubyte.gz', 'r')

# a = idx2numpy.convert_from_file(training_data)
# print(type(a[30000].reshape(-1)))


def main():
    #a = init_data(training_data)
    a = 3
    b = np.zeros(10)
    b[a]=1
    print(b)



def init_data(training_data):
    converted_data = idx2numpy.convert_from_file(training_data)
    converted_answers = idx2numpy.convert_from_file(training_answers)
    return converted_data, converted_answers


main()