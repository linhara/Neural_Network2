import numpy as np
import Layers
import random
import time
import idx2numpy
import gzip

training_data = gzip.open(r'C:\Users\linus\train-images-idx3-ubyte.gz', 'r')        #första 'r' gör att den inte letar efter t.ex \n
training_answers = gzip.open(r'C:\Users\linus\train-labels-idx1-ubyte.gz', 'r')
testing_data = gzip.open(r'C:\Users\linus\t10k-images-idx3-ubyte.gz', 'r')
testing_answers = gzip.open(r'C:\Users\linus\t10k-labels-idx1-ubyte.gz', 'r')

image_size = 28
num_images = 5

inp = np.array([1, 2, 3, 3, 4]).T
#structList = [len(inp), 5, 3]
structList = [784, 16, 16, 10]
iterations = 10000
layers = []
activation_of_each_layer = [inp]
learn_rate = 0.5
#expected = [0, 0, 1, 1, 1, 1, 0, 1, 0, 1]
# a = random.randint(0, (len(inputs)-1))

def main():
    start = time.time()
    init_struct()
    inputs, expected = init_data(training_data)             # EXPECTED ÄR EN INT, JÄMFÖRS MOT LEN 10 LISTA löser med np.zeroes

    rand = random.randint(0, (len(inputs[1]) - 1))

    for _ in range(iterations):

        run_network(inputs[rand].reshape(-1))       # reshape pga kan inte hantera matricies atm
        ans = np.zeros(10)
        ans[expected[rand]] = 1
        cost = calc_cost_prime(activation_of_each_layer[-1], ans)
        # -------------------------back_prop------------

        cor_output_layer(cost)
        for index_of_layer in reversed(range(len(layers)-1)):
            back_prop(index_of_layer)

    # -----------------Prints-----------------------------
    #print(inputs[a])
    print(expected[rand])
    print(activation_of_each_layer[-1])
    print(cost)
    print(time.time() - start)


# ------------Methods-------------------------------------
def init_struct():
    for i in range(len(structList[:-1])):
        layers.append(Layers.layer(structList[i], structList[i+1]))


def init_data(training_data):
    converted_data = idx2numpy.convert_from_file(training_data)
    converted_answers = idx2numpy.convert_from_file(training_answers)
    return converted_data, converted_answers


def run_network(inp):
    activations = inp
    for layer in layers:
        activations = layer.step_forward(activations)
        activation_of_each_layer.append(activations)        # do i really need this? memory waste


def back_prop(i):
    layers[i].error = layers[i].sig_prime() * np.dot(layers[i+1].error, layers[i+1].weights[:, 1:])
    hopefully_gradient = layers[i].received_activations[:1] * layers[i].error
    layers[i].weights += -learn_rate * hopefully_gradient[:, np.newaxis]


def cor_output_layer(cost):                                         # denna funktionen borde inte behövas
    layers[-1].error = cost
    hopefully_gradient = layers[-1].received_activations[:1] * cost
    layers[-1].weights += -learn_rate * hopefully_gradient[:, np.newaxis]


def calc_cost_prime(ans, expected):
    return 2*(ans - expected)


if __name__ == '__main__':
    main()
