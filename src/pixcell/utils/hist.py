class History:

    def __init__(self):
        self.epochs = []

    def add_epoch(
        self,
        epoch: int,
        logs: dict[str, float],
    ):
        self.epochs.append(
            {
                "epoch": epoch,
                **logs,
            }
        )

    def latest(self):
        return self.epochs[-1]

    def best(
        self,
        key: str,
        mode: str = "min",
    ):
        if mode == "min":
            return min(self.epochs, key=lambda x: x[key])

        if mode == "max":
            return max(self.epochs, key=lambda x: x[key])

        raise ValueError(mode)

    def state_dict(self):
        return {
            "epochs": self.epochs,
        }

    def load_state_dict(self, state):
        self.epochs = state["epochs"]