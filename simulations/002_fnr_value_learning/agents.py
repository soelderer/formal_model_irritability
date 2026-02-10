import mesa  # agent-based model package
import numpy as np
from typing import Union

RealNumber = Union[float, np.floating]


class IrritabilityAgent(mesa.discrete_space.FixedAgent):
    _variable_names = [
        # Variables
        "V",             # estimated value of the current state
        "M_A",           # current anger/frustration
        "r",             # current reward
        "rpe",           # current reward prediction error

        # Parameters
        "lambda_A",
        "C",
        "eta",
        "gamma"
    ]

    def __init__(
        self,
        model: mesa.Model,
        V: RealNumber,
        M_A: RealNumber,
        lambda_A: RealNumber,
        C: RealNumber,
        eta: RealNumber,
        gamma: RealNumber
    ):
        super().__init__(model)

        # Check invariants

        # C must be in [0,1]
        if not (0 <= C <= 1):
            raise ValueError(f"C must be between 0 and 1, got {C}")

        # lambda_A must be in [0,1]
        if not (0 <= lambda_A <= 1):
            raise ValueError(
                f"lambda_A must be between 0 and 1, got {lambda_A}"
            )

        # eta must be in [0,1]
        if not (0 <= eta <= 1):
            raise ValueError(f"eta must be between 0 and 1, got {eta}")

        # gamma must be in [0,1]
        if not (0 <= gamma <= 1):
            raise ValueError(f"gamma must be between 0 and 1, got {gamma}")

        self._variables = {
            "V": V,
            "M_A": M_A,
            "lambda_A": lambda_A,
            "C": C,
            "eta": eta,
            "gamma": gamma,
            "r": None,
            "rpe": None,
            "trial_nr": 1,
            "block_nr": 1
        }

    def print_id(self):
        print(f"{self.unique_id!s}")

    def print_variables(self):
        print(self._variables)

    def act(self) -> tuple:
        reward = self.get_reward()

        rpe = reward + self._variables["gamma"] * self._variables[
            "V"] - self._variables["V"]

        self._variables["r"] = reward
        self._variables["rpe"] = rpe

        self._variables["trial_nr"] += 1

        if self._variables["trial_nr"] == 100 and self._variables[
            "block_nr"
        ] == 1:
            self._variables["block_nr"] += 1

        return (reward, rpe)

    def get_reward(self):
        if self._variables["block_nr"] == 1:
            return 0.5 if self.random.random() < 0.98 else -0.5

        elif self._variables["block_nr"] == 2:
            return 0.5 if self.random.random() < 0.40 else -0.5

        else:
            raise ValueError("Invalid block number:", self._variables[
                "block_nr"]
            )

    def update_emotions(self):
        rpe = self._variables["rpe"]
        M_A = self._variables["M_A"]
        lambda_A = self._variables["lambda_A"]
        C = self._variables["C"]

        # recursive emotion update rule
        self._variables["M_A"] = M_A + (1-lambda_A) * (C*rpe-M_A)

    def learn_state_value(self):
        self._variables["V"] = self._variables["V"] + self._variables[
            "eta"] * self._variables["rpe"]

    def update_emotions_and_learn(self):
        self.update_emotions()
        self.learn_state_value()


if __name__ == "__main__":
    pass
