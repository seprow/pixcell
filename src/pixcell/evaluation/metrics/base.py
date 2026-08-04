import torch.nn as nn

class BaseMetric(nn.Module):

    def update(self, pred, target):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError