from algorithms.FFNetwork import FFNetwork

import numpy as np


def y_maps(n):
    res = np.zeros((10, 1))
    res[n][0] = 1
    return res

# utility functions for data pre-processing


def preprocess_data(trainX, trainY):
    processed = []
    for x, y in zip(trainX, trainY):
        flat_x = x.flatten() / 255.0
        processed.append((flat_x.reshape(flat_x.shape[0], 1), y_maps(y)))
    return processed


def main():
    trainX = testX = trainY = testY = None
    with np.load('datasets/mnist.npz') as data:
        lst = data.files
        trainX = data[lst[1]]
        trainY = data[lst[2]]

        testX = data[lst[0]]
        testY = data[lst[3]]

    train_data = preprocess_data(trainX, trainY)
    test_data = preprocess_data(testX, testY)

    p, q = train_data[0][0].shape, train_data[0][1].shape
    print(len(train_data))
    print(p, q)

    # It is very slow to train this on the CPU.
    network = FFNetwork([p[0], 100, q[0]], ['sigmoid', 'sigmoid'])
    network.fit_sgd(train_data, 0.5, 10, 30)

    correct = network.evaluate(test_data)
    print(f'Correct: {correct}%')

    FFNetwork.save(network, 'learned weights/handwritten_digits_02.json')


if __name__ == '__main__':
    main()
