import torch
import torch.nn.functional as F

from pixcell.evaluation.metrics.postprocessor import(
    BasePostProcessor,
    ComposePostProcessor
)

from pixcell.utils.registry import POSTPROCESSOR_REGISTRY


@POSTPROCESSOR_REGISTRY.register("identity")
class IdentityPostProcessor(BasePostProcessor):

    def __call__(self, pred, target):

        return pred, target


@POSTPROCESSOR_REGISTRY.register("sigmoid")
class SigmoidPostProcessor(BasePostProcessor):

    def __call__(self, pred, target):

        pred = torch.sigmoid(pred)

        return pred, target


@POSTPROCESSOR_REGISTRY.register("softmax")
class SoftmaxPostProcessor(BasePostProcessor):

    def __init__(self, dim=1):
        self.dim = dim

    def __call__(self, pred, target):

        pred = torch.softmax(pred, dim=self.dim)

        return pred, target


@POSTPROCESSOR_REGISTRY.register("threshold")
class ThresholdPostProcessor(BasePostProcessor):

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, pred, target):

        pred = (pred > self.threshold).float()

        return pred, target
    

@POSTPROCESSOR_REGISTRY.register("argmax")
class ArgmaxPostProcessor(BasePostProcessor):

    def __init__(self, dim=1, keepdim=True):
        self.dim = dim
        self.keepdim = keepdim

    def __call__(self, pred, target):

        pred = pred.argmax(dim=self.dim)

        return pred, target


@POSTPROCESSOR_REGISTRY.register("onehot")
class OneHotPostProcessor(BasePostProcessor):

    def __init__(self, num_classes=6, pred=True, target=True):
        self.num_classes = num_classes
        self.pred = pred
        self.target = target
        
    def __call__(self, pred, target):

        if self.pred:
            pred = F.one_hot(
                pred.long(),
                num_classes=self.num_classes,
            )

            pred = pred.movedim(-1, 1).float()

        if self.target:

            if target.shape[1] == 1:
                target = target.squeeze(1)
                
            target = F.one_hot(
                target.long(),
                num_classes=self.num_classes,
            )

            target = target.movedim(-1, 1).float()

        return pred, target
    

@POSTPROCESSOR_REGISTRY.register("multiclass_segmentation_dice")
def build_multiclass():

    return ComposePostProcessor([
        SoftmaxPostProcessor(),
        ArgmaxPostProcessor(),
        OneHotPostProcessor(target=False)
    ])


@POSTPROCESSOR_REGISTRY.register("binary")
def build_binary():

    return ComposePostProcessor([
        SigmoidPostProcessor(),
        ThresholdPostProcessor(),
    ])