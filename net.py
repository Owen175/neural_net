import math


class Layer:
    def __init__(self, num_inputs, num_outputs):
        self.biases = []
        self.weights = []
        self.inputs = num_inputs
        self.outputs = num_outputs

        for _ in range(num_inputs):
            self.weights.append([])
            for _ in range(num_outputs):
                self.weights[-1].append()
        # Can be referenced as self.weights[input_node_num][output_node_num] to find the weight.


class NN:
    def __init__(self, layer_dimensions):
        # Each node is connected to each in the next layer.
        self.layers = [Layer(layer_dimensions[i], layer_dimensions[i + 1]) for i in range(len(layer_dimensions) - 1)]

    def sigmoid_activation_function(self, value):
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

    def process_information(self, inputs):
        for layer in self.layers:
            inputs = self.calculate_layer(layer, inputs)
        return inputs


nn = NN([2, 2])
print(nn.process_information([1, 2]))