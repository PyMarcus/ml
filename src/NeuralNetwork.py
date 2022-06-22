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
