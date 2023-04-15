import numpy as np
import json

sigma = lambda z: 1.0/(1.0 + np.exp(-z))
sigma_prime = lambda z: sigma(z) * (1 - sigma(z))

relu = lambda z: max(0, z)
relu_prime = lambda z: 1 if z < 0 else 0 

activations = {
    'relu': (np.vectorize(relu), np.vectorize(relu_prime)),
    'sigmoid': (np.vectorize(sigma), np.vectorize(sigma_prime)),
}

cost_func = {
    'cross_entropy': {'func': lambda a, y: np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a))),
                      'delta': lambda z, a, y: (a - y)},
    'quadratic': {'func': lambda a, y: 0.5 * np.linalg.norm(a - y) ** 2,
                  'delta': lambda z, a, y: (a - y) * sigma_prime(z)},
}

"""
    Currently the algorithm only uses sigmoid. I will later implement
    a 'relu' and 'tanh' functions.
"""

class FFNetwork:
    def __init__(self, neuron_list, fs = None, cost='cross_entropy'):
        self.num_layers = len(neuron_list)
        self.neuron_sizes = neuron_list
        self.act_funcs = fs.copy() if fs is not None else ['sigmoid'] * (self.num_layers - 1)
        self.performance = 0

        self.cost_name = cost
        self.cost = cost_func[cost]

        """
            Remember, the cost function is not convex so weights and biases
            all being 0 may also be a local minima. So, it might get stuck
            at J(weights all 0 and biases all 0)
        """
        self.biases = [np.random.randn(x, 1) for x in neuron_list[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(
            neuron_list[1:], neuron_list[:-1])]

    """
        Assume that only the hidden and the output layers are defined.
    """
    def feed_forward(self, x):
        for l in range(self.num_layers - 1):
            act = activations[self.act_funcs[l]][0]
            w, b = self.weights[l], self.biases[l]

            x = act(w @ x + b)
        return x

    def acts_z(self, x):
        a = x
        outs, zs = [x], []
        for l in range(self.num_layers - 1):
            act = activations[self.act_funcs[l]][0]
            w, b = self.weights[l], self.biases[l]
            
            zs.append(w @ a + b)
            a = act(zs[-1])
            outs.append(a)

        return zs, outs

    def fit_bgd(self, train_data, eta, epochs):
        for e in range(epochs):
            np.random.shuffle(train_data)
            self.back_prop(train_data, eta)

            print(f'Epoch: {e + 1} complete')

    def fit_sgd(self, train_data, eta, mini_batch_size, epochs):
        n = len(train_data)

        for e in range(epochs):
            np.random.shuffle(train_data)
            mini_batches = [train_data[k: k + mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            ex = 0
            for batch in mini_batches:
                self.back_prop(batch, eta)
                if ex % np.floor(len(mini_batches) * 0.10) == 0:
                    # print(f'Batches: {ex}')
                    pass
                ex += 1

            print(f'Epoch: {e + 1} complete | {ex}')

    def back_prop(self, batch, eta):
        m = len(batch)

        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        for x, y in batch:
            delta_w = [np.zeros(w.shape) for w in self.weights]
            delta_b = [np.zeros(b.shape) for b in self.biases]
            zs, acts = self.acts_z(x)

            delta = self.cost['delta'](zs[-1], acts[-1], y)

            delta_b[-1] = delta
            delta_w[-1] = delta @ acts[-2].T
            for l in range(2, self.num_layers):
                z = zs[-l]

                sp = activations[self.act_funcs[-l]][1](z)
                delta = self.weights[-l+1].T @ delta * sp
                del_w = delta @ acts[-l-1].T
                delta_b[-l] = delta
                delta_w[-l] = del_w

            nabla_b = [nb + db for nb, db in zip(nabla_b, delta_b)]
            nabla_w = [nw + dw for nw, dw in zip(nabla_w, delta_w)]

        self.weights = [w - (eta / m) * nabla for w,
                        nabla in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / m) * nabla for b,
                       nabla in zip(self.biases, nabla_b)]

    def evaluate(self, test_data):
        self.performance = sum([int(np.argmax(self.feed_forward(row[0])) == np.argmax(
            row[1])) for row in test_data]) / len(test_data) * 100
        return self.performance



    @staticmethod
    def save(network, path):
        # for w, b in zip(self.weights, self.biases):
        # Weights and biases are stored separated by '\n$$\n'
        data = {'sizes': network.neuron_sizes,
                'act_funcs' : network.act_funcs,
                'cost_func': network.cost_name,
                'performance': str(network.performance),
                'biases': [b.tolist() for b in network.biases],
                'weights': [w.tolist() for w in network.weights], }
        with open(path, "w") as f:
            json.dump(data, f)

    @staticmethod
    def load(path):
        with open(path, 'r') as f:
            data = json.load(f)

        network = FFNetwork(data['sizes'], data['act_funcs'], data['cost_func'])
        network.performance = data['performance']
        network.biases = [np.asarray(b) for b in data['biases']]
        network.weights = [np.asarray(w) for w in data['weights']]

        return network
