import numpy as np
import matplotlib.pyplot as plt
import math as m
import random
import pandas as pd


learning_rate = 0.01
lmbda = 0.01
beta = 0.09
layers_dims = []


def sin_problem():
    data = pd.read_csv("sin_data.csv")

    ann = ANN([1,25,1])

    train, val, test = ann.split_dataset(data)
    input_layer, output_layer = pd.DataFrame(train["x"]), pd.DataFrame(train["sin_x"])
    val_ip, val_op = pd.DataFrame(val["x"]), pd.DataFrame(val["sin_x"])
    test_ip, test_op = pd.DataFrame(test["x"]), pd.DataFrame(test["sin_x"])
    parameters = ann.train_model(input_layer, output_layer, epochs=8000)

    print(parameters["b1"], parameters["b2"])
    all_out = np.array([])
    all_in = np.array([])

    hidden_layer = ann.tanhx(np.dot(parameters["W1"],input_layer.T) + parameters["b1"])
    output_layer = ann.tanhx(np.dot(parameters["W2"],hidden_layer) + parameters["b2"])

    all_out = np.append(all_out, output_layer)
    all_in = np.append(all_in, input_layer)

    plt.plot(train["x"],train["sin_x"],'r')
    plt.plot(all_in,all_out,'b')
    plt.show()


class ANN(object):
    def __init__(self, layers_dims, learning_rate=0.1, lmbda=0.01, beta=0.09):
        self.parameters = {}
        self.grads = {}
        self.weights = {}
        self.layers_dims = layers_dims
        self.train_cost = []
        self.val_cost = []
        self.test_cost = []
        self.reg = []
        self.momentum = []
        self.learning_rate = learning_rate
        self.lmbda = lmbda
        self.beta = beta

    def tanhx(self, data):
        output = np.tanh(data)
        return output

    def tanhx_der(self, data):
        return 1 - self.tanhx(data)**2

    def io_normalization(self, data):
        for i in data.columns:
            data[i] = (2*data[i] - data.max()[i] - data.min()[i])/(data.max()[i] - data.min()[i])
        return data

    def initialize_parameters(self, layer_dims):
        for i in range(1,len(layer_dims)):
            self.parameters["W" + str(i)] = np.random.randn(layer_dims[i],layer_dims[i-1])*0.2
            self.parameters["b" + str(i)] = np.zeros(shape=(layer_dims[i],1))
            self.weights["W" + str(i)] = [self.parameters["W" + str(i)].mean()]


        return self.parameters


    def feedforward(self, input_layer, parameters):
        activations = [input_layer]
        L = len(parameters)//2
        for i in range(1, L):
            activation = self.tanhx(np.dot(parameters["W" + str(i)], activations[i-1]) + parameters["b" + str(i)])
            activations += [activation]

        output = self.tanhx(np.dot(parameters["W" + str(L)],activations[L-1]) + parameters["b" + str(L)])
        activations += [output]
        print("ff done")
        return output, activations

    def cost_function(self, predictions, outputs):
        print("found cost")
        self.train_cost = (1/1000)*(outputs-predictions)**2/2
        return (1/1000)*(outputs-predictions)**2/2

    def cost_validation(self, predictions, outputs):
        print("found cost")
        self.val_cost = (1/1000)*((outputs-predictions)**2/2)
        return (1/1000)*((outputs-predictions)**2/2)

    def backprop(self, actual_outputs, outputs, activation_cache):
        L = len(self.parameters)//2
        self.grads["dZ" + str(L)] = outputs - actual_outputs
        self.grads["dW" + str(L)] = (1/1000)*np.matmul(self.grads["dZ" + str(L)], activation_cache[L-1].T)
        self.grads["db" + str(L)] = (1/1000)*np.sum(self.grads["dZ" + str(L)], axis = 1, keepdims = True)

        for layer in range(L-1, 0, -1):
            self.grads["dZ" + str(layer)] = np.multiply(np.dot(self.parameters["W" + str(layer+1)].T, self.grads["dZ" + str(layer+1)]), self.tanhx_der(activation_cache[layer]))
            self.grads["dW" + str(layer)] = (1/1000)*np.matmul(self.grads["dZ" + str(layer)], activation_cache[layer - 1].T)
            self.grads["db" + str(layer)] = (1/1000)*np.sum(self.grads["dZ" + str(layer)], axis = 1, keepdims=True)

        print("backprop done")
        return self.grads

    def regularization(self, grad_comp, param_comp):
        dW = learning_rate*(grad_comp - lmbda*abs(param_comp))
        return dW

    def momentum_term(self, grad_comp, param_comp):
        dW = beta*grad_comp + (beta-1)*(self.regularization(grad_comp, param_comp))
        return dW

    def update_parameters(self, learning_rate):
        """
        Takes care of regularization and momentum too
        """
        for i in range(0, len(self.parameters)//2):
            #print(i)
            #dW = self.momentum_term(self.grads["dW" + str(i+1)], self.parameters["W" + str(i+1)])
            self.parameters["W" + str(i+1)] = self.parameters["W" + str(i+1)] - learning_rate*self.grads['dW' + str(i+1)]
            self.parameters["b" + str(i+1)] = self.parameters["b" + str(i+1)] - learning_rate*self.grads["db" + str(i+1)]
            self.weights["W" + str(i+1)] = self.weights["W" + str(i+1)] + [self.parameters["W" + str(i+1)].mean()]
        print("up done")
        return self.parameters


    def train_model(self, input, output, epochs):
        parameters = self.initialize_parameters(self.layers_dims)

        actual_output = output.values.reshape(self.layers_dims[-1], output.shape[0])
        inputs = input.values.reshape(self.layers_dims[0], input.shape[0])

        for epoch in range(epochs):
            '''random.shuffle(training_data)
            print(type(training_data[0]), training_data[0])
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            '''
            outputs, activation_cache = self.feedforward(inputs, parameters)

            cost = self.cost_function(outputs, actual_output)

            grads = self.backprop(actual_output, outputs, activation_cache)

            parameters = self.update_parameters(learning_rate)

            output = "Epoch %r Error %r"%(epoch+1, cost.mean())
            print(output)
            
        self.plot_weights_over_epochs()
        return parameters

    def split_dataset(self, data):
        actual = data
        data_indices = [i for i in range(len(data))]
        test_indices = random.sample(range(len(data)), int(0.1*len(data)))
        data_indices = np.delete(data_indices, test_indices)

        train = []
        val = []
        test = []
        data = data.values.tolist()
        for i in range(0,len(data)):
            if(i in data_indices):
                train += [data[i]]
            else:
                test += [data[i]]
        train = []
        for i in range(0, len(data_indices)):
            if(i%5==0):
                val += [data[i]]
            else:
                train += [data[i]]

        train = pd.DataFrame(train, columns = actual.columns)
        val = pd.DataFrame(val, columns = actual.columns)
        test = pd.DataFrame(test, columns = actual.columns)
        return train, val, test

    def rms_error(self, outputs, predictions):
        ms_sum = 0
        for op, pred in zip(outputs, predictions):
            ms_sum += (output[i] - predictions[i])**2
        rmse = ms_sum/len(outputs)
        return rmse
    
    def plot_weights_over_epochs(self):
        for i in range(0, len(self.parameters)//2):
            plt.title("W" + str(i+1) +" over epochs")
            plt.xlabel("Epochs")
            plt.ylabel("Weights")
            plt.plot(self.weights["W" + str(i+1)])
            plt.show()


if(__name__ == "__main__"):


    sin_problem()
    '''data = pd.read_excel("uci.xlsx")

    ann = ANN([4,20,1])
    data = ann.io_normalization(data)
    print("Normalization Done")
    #print(data, type(data))

    train, val, test = ann.split_dataset(data)

    train_x, train_y = pd.DataFrame(train[["AT","V","AP","RH"]]), pd.DataFrame([train[["PE"]]])
    val_x, val_y = pd.DataFrame(val[["AT","V","AP","RH"]]), pd.DataFrame([val[["PE"]]])
    test_x, test_y = pd.DataFrame(test[["AT","V","AP","RH"]]), pd.DataFrame([test[["PE"]]])

    print(train_x, len(train_x))
    parameters = ann.train_model(train_x, train_y, epochs = 1)

    hidden_layer = ann.tanhx(np.dot(parameters["W1"],input_layer.T) + parameters["b1"])
    output_layer = ann.tanhx(np.dot(parameters["W2"],hidden_layer) + parameters["b2"])

    print(output_layer)
    all_out = np.append(all_out, output_layer)
    all_in = np.append(all_in, input_layer)

    print(rms_error(train_y, output_layer))'''
