import inspect

class TorchSchedulerWrapper:

    def __init__(self, scheduler_cls, optimizer, **params):
        self.scheduler = scheduler_cls(optimizer, **params)

        self._step_signature = inspect.signature(
            self.scheduler.step
        )

    def step(self, **kwargs):

        valid_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in self._step_signature.parameters
        }

        return self.scheduler.step(**valid_kwargs)

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state):
        return self.scheduler.load_state_dict(state)