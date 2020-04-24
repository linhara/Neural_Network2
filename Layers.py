import numpy as np
class layer:
    def __init__(self, nr_of_neurons, weights_per_neuron):
        self.weights = np.random.randn(weights_per_neuron, nr_of_neurons + 1)   # the "+1" is to add another coloumn acting as the biases
        self.received_activations = None
        self.activations = None
        self.error = None

    def step_forward(self, activations):
        self.received_activations = activations
        Z = np.dot(self.weights, (np.hstack(([1], activations))))        # adding a one at the beginning of vector so biases is constant
        self.activations = self.sigmoid(Z)
        return self.activations

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sig_prime(self):
        return self.activations * (1 - self.activations)

    #def back_prop(self, next_layer_error, next_layer_weights, learn_rate):
        #self.error = self.sig_prime() * np.dot(next_layer_error, next_layer_weights)
        #hopefully_gradient = self.received_activations * self.error
        #self.weights += -learn_rate * hopefully_gradient