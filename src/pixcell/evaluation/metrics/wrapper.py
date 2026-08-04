from pixcell.evaluation.metrics import BaseMetric


class TorchMetricWrapper(BaseMetric):

    def __init__(self, metric, postprocessor=None):
        super().__init__()
        self.metric = metric
        self.postprocessor = postprocessor

    def update(self, pred, target):

        if self.postprocessor:
            pred, target = self.postprocessor(pred, target)

        self.metric.update(pred, target)

    def compute(self):
        return self.metric.compute()

    def reset(self):
        self.metric.reset()


class MonaiMetricWrapper(BaseMetric):

    def __init__(self, metric, postprocessor=None):
        super().__init__()
        self.metric = metric
        self.postprocessor = postprocessor

    def update(self, pred, target):

        if self.postprocessor:
            pred, target = self.postprocessor(pred, target)

        self.metric(y_pred=pred, y=target)

    def compute(self):
        return self.metric.aggregate()

    def reset(self):
        self.metric.reset()