import torch
from loss_function import LossFunction

class MyRootMeanSquare(LossFunction):
    def add(self, predict, target):
        self.loss_value += ((predict - target) ** 2)
        self.count += 1

    def calculate_average_loss(self):
        # sqrt(1/N * sum(loss))
        return torch.sqrt(self.loss_value * (1 / self.count))