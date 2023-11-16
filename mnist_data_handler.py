from mnist import MNIST

mndata = MNIST('data')


def get_training_data():
    return mndata.load_training()
# Returns in the form images, labels

def get_testing_data():
    return mndata.load_testing()