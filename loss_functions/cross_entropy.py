import torch
from loss_function import LossFunction

class MyCrossEntropy(LossFunction):
    def add(self, predict, target):
        epsilon = 1e-10
        self.loss_value += (target * torch.log(predict + epsilon) + (1 - target) * torch.log(1 - predict + epsilon))
        self.count += 1

    def calculate_average_loss(self):
        # -1/N * sum(loss)
        return (-1 / self.count) * self.loss_value