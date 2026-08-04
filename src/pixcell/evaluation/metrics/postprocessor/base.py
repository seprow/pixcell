from abc import ABC, abstractmethod


class BasePostProcessor(ABC):

    @abstractmethod
    def __call__(self, pred, target):
        """
        Returns
        -------
        pred, target
        """
        pass


