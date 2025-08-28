class LossFunction:
    def __init__(self):
        self.loss_value = 0 # sum of all losses
        self.count = 0 # number of samples

    def add(self, predict, target):
        raise NotImplementedError("Subclasses should implement this method")

    def calculate_average_loss(self):
        raise NotImplementedError("Subclasses should implement this method")