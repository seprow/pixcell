# $env:PYTHONPATH="src" 

from pathlib import Path

from pixcell.runners import SupervisedRunner
from pixcell.utils import load_yaml 
from pixcell.configs import Config

YAML_PATH = Path(r".\configs\default_config.yaml") 

def main():

    config = Config.from_dict(
        load_yaml(YAML_PATH)
    )

    if config.training.learning_mode == 'supervised':

        SupervisedRunner(config).run()

    elif config.training.learning_mode == 'semi_supervised':

        raise NotImplementedError(
            'Semi-supervised runner is not implemented yet.'
        )

    elif config.training.learning_mode == 'unsupervised':

        raise NotImplementedError(
            'Unsupervised runner is not implemented yet.'
        )

    else:

        raise ValueError(
            f'Unknown learning mode: {config.training.learning_mode}'
        )


if __name__ == '__main__':
    main()
