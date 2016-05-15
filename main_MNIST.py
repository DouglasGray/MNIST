import NN
import mnist_loader

def load_data():
    train, val, test = mnist_loader.load_data_wrapper()
    train = list(train)
    val = list(val)
    test = list(test)

    return train, val, test

if __name__ == "__main__":
    training_data, validation_data, test_data = load_data()

    # Set network parameters and cost/transfer functions
    layers = [784, 30, 10]
    lmbda = 5.0
    eta = 0.5
    epochs = 30
    nbatch = 10

    cost = NN.CrossEntropy
    transfer_fns = [NN.sigmoid, NN.sigmoid]

    # Create the network, train and test
    nn = NN.Network(layers, cost, transfer_fns)
    nn.train_network(training_data, validation_data, epochs, nbatch, eta, lmbda)

    acc = nn.test_network(test_data)
    print("Accuracy on test data: {} / {})".format(acc, len(test_data)))
