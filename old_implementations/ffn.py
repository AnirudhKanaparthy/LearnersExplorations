import numpy as np

# This is XOR data

trainX = np.asarray(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],

    ]
)
trainY = np.asarray(
    [
        [1, 0],
        [0, 1],
        [0, 1],
        [1, 0],

    ]
)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu_normal(z):
    return max(0, z)

relu = np.vectorize(relu_normal)

# We only store the layers from 1 to n
# n is the last layer index.
def init_network(input_size, layers: list):
    weights = [np.zeros((layers[0], input_size))]
    biases = [np.zeros((layers[0], 1))]
    for i in range(1, len(layers)):
        weights.append(np.zeros((layers[i], layers[i - 1])))
        biases.append(np.zeros((layers[i], 1)))
    return {'W': weights, 'b': biases}

def forward_prop(network, x: np.ndarray):
    weights, biases = network['W'], network['b']
    act = x
    for i in range(len(weights)):
        act = sigmoid(weights[i] @ act + biases[i])
    return act

def layer_activations(network, x: np.ndarray):
    weights, biases = network['W'], network['b']
    activations = [x]
    for i in range(len(weights)):
        activations.append(sigmoid(weights[i] @ activations[i] + biases[i]))
    return activations


# This is assuming sigmoid
def back_prop(network, trainX, trainY):
    l = len(network['W']) - 1
    alpha = 1
    for _ in range(1000):
        for i in range(len(trainX)):
            acts = layer_activations(network, trainX[i].reshape(len(trainX[i]), 1))
            d_c_a = 2 * (acts[-1] - trainY[i].reshape(len(trainY[i]), 1))

            for j in range(l + 1):
                z = network['W'][l - j] @ acts[l - j] + network['b'][l - j]
                sig = sigmoid(z)

                d_a_z = sig * (1 - sig)
                d_z_w = np.repeat(acts[l - j].T, d_a_z.shape[0], axis=0)

                network['W'][l - j] -= alpha * d_z_w * d_a_z * d_c_a
                network['b'][l - j] -= alpha * d_a_z * d_c_a

                
                # print(f'Z:\n{z}\n\nd_a_z:\n{d_a_z}\n\nd_c_a:\n{d_c_a}\n\nd_z_w:\n{d_z_w}\n\n')
                # wgt, bias = network['W'][l - j], network['b'][l - j]
                # print(f'Weight:\n{wgt}\n\nBias:\n{bias}\n\n')
                
                d_c_a = np.sum((network['W'][l - j] * d_a_z * d_c_a).T, axis=1).reshape(network['W'][l - j].shape[1], 1)
    return network




def main():
    network = init_network(2, [3, 2, 2])
    back_prop(network, trainX, trainY)

    print('________________________\n')
    for i in range(len(network['W'])):
        weight, bias = network['W'][i], network['b'][i]
        print(f'Weights {i}:\n{weight}\n\nBiases {i}:\n{bias}\n')
    print('________________________\n')

    for x in trainX:
        res = forward_prop(network, x.reshape(2, 1))
        print(f'Result:\n{res}')

if __name__ == '__main__':
    main()