from algorithms.FFNetwork import FFNetwork
import numpy as np

def preprocess_data(data):
    processed = []
    for x, y in data:
        vec_x = np.asarray(x).reshape(len(x), 1)
        vec_y = np.asarray(y).reshape(len(y), 1)
        processed.append((vec_x, vec_y))
    return processed


def main():
    # XOR data
    raw_data = [
        ([0, 0], [1, 0]),
        ([0, 1], [0, 1]),
        ([1, 0], [0, 1]),
        ([1, 1], [1, 0]),

        ([0, 0], [1, 0]),
        ([0, 1], [0, 1]),
        ([1, 0], [0, 1]),
        ([1, 1], [1, 0]),

        ([0, 0], [1, 0]),
        ([0, 1], [0, 1]),
        ([1, 0], [0, 1]),
        ([1, 1], [1, 0]),
    ]
    train_data = preprocess_data(raw_data)
    

    network = FFNetwork([2, 3, 2])

    # If it doesn't fit properly then it just means that it found a local minima
    
    network.fit_sgd(train_data, 0.1, 3, 5000)
    # network.fit_bgd(train_data, 0.1, 1000)

    for x, y in train_data:
        res = network.feed_forward(x)
        print('______________________')
        print(f'{x.T} => {res.T} || {y.T}')
        print('______________________\n\n')