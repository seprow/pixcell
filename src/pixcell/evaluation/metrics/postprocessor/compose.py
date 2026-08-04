from pixcell.evaluation.metrics.postprocessor import BasePostProcessor

class ComposePostProcessor(BasePostProcessor):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, pred, target):

        for t in self.transforms:
            pred, target = t(pred, target)

        return pred, target