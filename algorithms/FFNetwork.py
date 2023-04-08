import numpy as np
import json

activations = {
    'relu': np.vectorize(lambda z: max(0, z)),
    'sigmoid': np.vectorize(lambda z: 1.0/(1.0 + np.exp(-z))),
}

"""
    Currently the algorithm only uses sigmoid. I will later implement
    a 'relu' and 'tanh' functions.
"""


class FFNetwork:
    def __init__(self, neuron_list, act='sigmoid'):
        self.num_layers = len(neuron_list)
        self.neuron_sizes = neuron_list
        self.act = activations[act]
        self.performance = 0

        """
            Remember, the cost function is not convex so weights and biases
            all being 0 may also be a local minima. So, it might get stuck
            at J(weights all 0 and biases all 0)
        """
        self.biases = [np.random.randn(x, 1) for x in neuron_list[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(
            neuron_list[1:], neuron_list[:-1])]

    def feed_forward(self, x):
        act = self.act
        for w, b in zip(self.weights, self.biases):
            x = act(w @ x + b)
        return x

    def acts_z(self, x):
        a = x
        acts = [x]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = w @ a + b
            zs.append(z)

            a = self.act(z)
            acts.append(a)

        return zs, acts

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
                    print(f'Batches: {ex}')
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

            cost_gradient = acts[-1] - y
            sp = self.sigma_prime(zs[-1])
            delta = cost_gradient * sp

            del_w = delta @ acts[-2].T
            delta_b[-1] = delta
            delta_w[-1] = del_w
            for l in range(2, self.num_layers):
                z = zs[-l]

                sp = self.sigma_prime(z)
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

    def sigma_prime(self, z):
        s = self.act(z)
        return s * (1 - s)

    @staticmethod
    def save(network, path):
        # for w, b in zip(self.weights, self.biases):
        # Weights and biases are stored separated by '\n$$\n'
        data = {'sizes': network.neuron_sizes,
                'performance': str(network.performance),
                'biases': [b.tolist() for b in network.biases],
                'weights': [w.tolist() for w in network.weights], }
        with open(path, "w") as f:
            json.dump(data, f)

    @staticmethod
    def load(path):
        with open(path, 'r') as f:
            data = json.load(f)

        network = FFNetwork(data['sizes'])
        network.performance = data['performance']
        network.biases = [np.asarray(b) for b in data['biases']]
        network.weights = [np.asarray(w) for w in data['weights']]

        return network
