
# idea for behind of a perceptron (neural web)
import random
import numpy as np


# amateur version :)
class Perceptron:
    def __init__(self, no_of_inputs):
        self.no_of_inputs = no_of_inputs

    def perceptron(self):
        return list(range(len(self.no_of_inputs)))

    def scalar(self) -> int:
        """
        scalar product
        :return: int
        """
        result = []
        z = zip(self.no_of_inputs, self.perceptron())
        [result.append(x * y) for x, y in list(z)]
        return random.randint(1, 10000) + sum(result)


# better version :0
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


if __name__ == '__main__':
    neural_of_computer = Perceptron_([1, 2, 3, 4])
    value = neural_of_computer.predict()
    print(value)
