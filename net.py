import math
import random


class Layer:
    def __init__(self, num_inputs, num_outputs):
        self.biases = []
        self.bias_gradient = []
        for i in range(num_outputs):
            self.biases.append(0)
            self.bias_gradient.append(0)
        self.weights = []
        self.weight_gradient = []
        for i in range(num_inputs):
            self.weights.append([])
            self.weight_gradient.append([])
            for j in range(num_outputs):
                self.weights[-1].append(random.randint(-1000, 1000)/1000)
                self.weight_gradient[-1].append(0)

        self.inputs = num_inputs
        self.outputs = num_outputs

        # Can be referenced as self.weights[input_node_num][output_node_num] to find the weight.


class NN:
    def __init__(self, layer_dimensions):
        # Each node is connected to each in the next layer.
        self.layers = [Layer(layer_dimensions[i], layer_dimensions[i + 1]) for i in range(len(layer_dimensions) - 1)]

    def sigmoid_activation_function(self, value):
        if value < -100:
            value = -100
        elif value > 100:
            value = 100
        return 1/(1+math.e**(-value))

    def calculate_layer(self, layer, inputs):
        processed_outputs = []
        for i, bias in enumerate(layer.biases):
            node_total = 0
            for j in range(layer.inputs):
                node_total += layer.weights[j][i] * inputs[j]
            node_total += bias
            processed_outputs.append(self.sigmoid_activation_function(node_total))
        return processed_outputs

    def testing(self, data):
        inputs = data.inputs
        for layer in self.layers:
            inputs = self.calculate_layer(layer, inputs)
        return inputs

    def training(self, data, learnrate):
        self.calculate_gradients(learnrate, data)

    def get_cost(self, data):
        outputs = self.testing(data)
        cost = 0
        try:
            for output, expected_output in zip(outputs, data.prediction):
                cost += (output - expected_output)**2
        except Exception as e:
            print(e, outputs, data.prediction)
            exit()
        return cost

    def individual_node_cost_derivative(self, output, expected_output):
        return 2 * (output - expected_output)

    def sigmoid_activation_derivative(self, input):
        activation = self.sigmoid_activation_function(input)
        return activation * (1-activation)



    def update_weights_and_biases(self, layer, learn_rate):
        for i in range(layer.outputs):
            layer.biases[i] -= layer.bias_gradient[i] * learn_rate
            for j in range(layer.inputs):
                layer.weights[j][i] = layer.weight_gradient[j][i] * learn_rate

    def calculate_gradients(self, learnrate, data):
        h = 0.000001
        cost = self.get_cost(data)

        for layer in self.layers:
            for i in range(layer.inputs):
                for j in range(layer.outputs):
                    layer.weights[i][j] += h
                    change_in_cost = self.get_cost(data) - cost
                    layer.weights[i][j] -= h
                    layer.weight_gradient[i][j] = change_in_cost / h
            for i in range(layer.outputs):
                layer.biases[i] += h
                change_in_cost = self.get_cost(data) - cost
                layer.biases[i] -= h
                layer.bias_gradient[i] = change_in_cost / h

        for layer in self.layers:
            self.update_weights_and_biases(layer, learnrate)

# data = Training_Data([1, 2], [0, 1])
# nn = NN([2, 20, 2])
# nn.get_cost(data)
# for i in range(1000):
#     nn.training(data, 20)
#
# print(nn.testing(data))
