from loguru import logger

class Registry:

    def __init__(self):
        self._objects = {}

    def register(self, name: str):

        def decorator(obj):

            if name in self._objects: 
                
                logger.warning(
                    f"'{name}' is already registered. Overwriting the previous registration."
                )

            self._objects[name] = obj

            return obj

        return decorator

    def get_class(self, name: str):

        try:
            return self._objects[name]

        except KeyError:
            raise ValueError(
                f"{name} is not registered."
            )

    def build(
        self,
        name: str,
        **kwargs,
    ):
        return self.get_class(name)(**kwargs)

    def registered(self):

        return tuple(sorted(self._objects))

# Data
IMAGE_READER_REGISTRY = Registry()
ANNOTATION_BUILDER_REGISTRY = Registry()
RESAMPLE_PIPELINE = Registry()
TRANSFORM_PIPELINE = Registry()

# Trainig
LOSS_REGISTRY = Registry()
LR_SCHEDULERS = Registry()
POSTPROCESSOR_REGISTRY = Registry()
METRIC_REGISTRY = Registry()

# model
MODEL_REGISTRY = Registry()



