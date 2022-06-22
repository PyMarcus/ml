# Machine learning to solve linear regression


### Perceptron
In machine learning, the perceptron is an algorithm for supervised learning of binary classifiers. A binary classifier is a function which can decide whether or not an input, represented by a vector of numbers, belongs to some specific class. It is a type of linear classifier, i.e. a classification algorithm that makes its predictions based on a linear predictor function combining a set of weights with the feature vector.<br>
<img src="/assets/perceptron.png">
<br>

    import numpy as np


    class Perceptron_:

        def __init__(self, num_input):
            """
            Perceptron
            :param num_input: 
            """
            self.weights = np.zeros(len(num_input) + 1)
            self.inputs = num_input
    
        def predict(self) -> int:
            activation = 0
            summation = np.dot(self.inputs, self.weights[1:]) + self.weights[0]  # scalar product + bayers number
            if summation > 0: # there is part of graphic
                return summation
            return activation

<br>

### Neural network

An ANN is based on a collection of connected units or nodes called artificial neurons, which loosely model the neurons in a biological brain. Each connection, like the synapses in a biological brain, can transmit a signal to other neurons. An artificial neuron receives signals then processes them and can signal neurons connected to it. The "signal" at a connection is a real number, and the output of each neuron is computed by some non-linear function of the sum of its inputs. The connections are called edges. Neurons and edges typically have a weight that adjusts as learning proceeds.
<br>
<img src="/assets/network.svg.png">

    from torch import nn


    class NeuralNetwork(nn.Module):
        """
        Its a set of perceptrons that
        represents a function (its not
        a deep neural network ,neural network with many layers).
        """
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.Sequential(nn.Linear(1, 1))  # linear operation that receive a weight and returns a value
            ...

    def forward(self, data: int) -> nn.Sequential:
        return self.layers(data)