from mnist_data_handler import get_training_data, get_testing_data
from net import NN
training_images, training_labels = get_training_data()
class Data:
    def __init__(self, inputs, prediction):
        self.inputs = inputs
        temp = prediction
        self.prediction = [0] * 9
        self.prediction.insert(temp, 1)

data_list = []

training_images, training_labels = get_training_data()
for image, label in zip(training_images, training_labels):
    data_list.append(Data(image, label))

nn = NN([784, 200, 10])
nn.get_cost(data_list[0])
count = 0
for data in data_list:
    nn.training(data, 20)
    print(count)
    count += 1
    if count == 10:
        print(nn.get_cost(data))
print(nn.get_cost(data_list[0]))