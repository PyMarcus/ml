from typing import Any
import torch.distributions.uniform as tdu
import torch.optim
from torch.utils.data import Dataset, DataLoader
from torch import nn


def lossFunction():
    """
    Based in Mean squared error, can solve the
    approximation of the true response.
    This function is a metric for find the result.

    Futthermore, this function check if the response was wrong,
    trought of distance in between 2 points (the line if u want and the error function)
    :return:
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"  # check is nvidea card available
    print(f"Running at {device}")
    model = LineNetwork().to(device)
    loss = nn.MSELoss() # indentify the error
    # (stochastic gradient descent) solve the gradient to find the smaller point
    optimizer = torch.optim.SGD(model.parameters(), lr=le-3) # model to optimize and learning rate



class TrainingAneuralNetwork(Dataset):
    """To train a neural network, samples are needed"""
    def __init__(self, which_function, interval, number_of_samples) -> None:
        """
        :param which_function: funtion to train
        :param interval:  interval of data
        :param number_of_samples: total of samples
        """

        # begin and finish of interval
        # and sample the amount of samples
        self.points_choiceds = tdu.Uniform(interval[0], interval[1]).sample(number_of_samples)
        print(f"Uniform points choiceds from interval: {self.points_choiceds}")
        self.data_set = [(data, which_function(data) for data in self.points_choiceds)]  # apply the function in which data
        print(f"Set of data how result from data generate and function apply in each one: {self.data_set}")
        ...

    def __len(self) -> int:
        """
        Length of dataset
        :return: int
        """
        return len(self.data_set)

    # @override
    def __getitem__(self, item) -> Any:
        """
        Get a item from dataset
        :param item:
        :return:
        """
        return self.data_set[item]


if __name__ == '__main__':
    which_function = lambda x : 2 * x + 3  # 2x + 3
    interval = (-10, 10)
    samples = 1000
    sample_test = 100
    train_neural = TrainingAneuralNetwork(which_function, interval, samples)
    train_neural_test = TrainingAneuralNetwork(which_function, interval, sample_test)


    # loader data during the train
    train_dataloader_test = DataLoader(train_neural_test,batch_size=train_neural_test ,shuffle=True)  # data to loader
    # size of data to loader, shuffle, yes!

    train_dataloader = DataLoader(train_neural, batch_size=train_neural, shuffle=True)
