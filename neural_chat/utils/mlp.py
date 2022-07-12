from typing import List
from torch import nn
from neural_chat.logger import Loggable


class MLP(nn.Module, Loggable):
    """Defines a standard Multi Layer Perceptron"""

    def __init__(self, input_dim: int, hidden_dim: List[int], output_dim: int) -> None:
        nn.Module.__init__(self)
        Loggable.__init__(self)

        # create mlp
        sizes = [input_dim] + hidden_dim + [output_dim]
        mods = sum(
            [
                [nn.Linear(i, j), nn.ReLU(inplace=True)]
                for i, j in zip(sizes, sizes[1:])
            ],
            [],
        )
        mods = mods[:-1]

        self.mlp = nn.Sequential(*mods)

        self._mlp_hyperparams = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dim": hidden_dim,
        }

    def forward(self, x):
        return self.mlp.forward(x)

    def log_local_hyperparams(self):
        return self._mlp_hyperparams

    def log_local_epoch(self):
        return self.mlp.state_dict()
