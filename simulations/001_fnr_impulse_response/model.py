import mesa
import numpy as np
import seaborn as sns
from typing import Dict


class IrritabilityAgent(mesa.Agent):
    def __init__(self, model, init_variables: Dict, init_parameters: Dict):
        super().__init__(model)

        # TODO: check for invariants here (e.g. some parameters must be
        # in [0,1])

        self.init_variables_ = init_variables
        self.init_parameters_ = init_parameters


if __name__ == "__main__":
    pass
