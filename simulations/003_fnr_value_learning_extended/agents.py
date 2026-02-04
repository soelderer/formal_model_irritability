import mesa  # agent-based model package
from typing import Dict
from scipy.special import softmax
from enum import Enum


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
        "gamma",
        "alpha",
        "kappa"
    ]

    def __init__(
        self,
        model,
        V=None,
        M_A=None,
        r=None,
        rpe=None,
        lambda_A=None,
        C=None,
        eta=None,
        gamma=None,
        alpha=None,
        kappa=None
    ):
        super().__init__(model)

        # TODO: check if init_variable.keys() match _variable_names

        # TODO: check for invariants here (e.g. some parameters must be
        # in [0,1])

        # raise ValueError()

        self._variables = {
            "V": V,
            "M_A": M_A,
            "r": r,
            "rpe": rpe,
            "lambda_A": lambda_A,
            "C": C,
            "eta": eta,
            "gamma": gamma,
            "alpha": alpha,
            "kappa": kappa,
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

    def negativity_bias(self, affective_input):
        return (
            affective_input
            if affective_input > 0
            else self._variables["kappa"] * affective_input
        )

    def update_emotions_and_learn(self):
        r = self._variables["r"]
        rpe = self._variables["rpe"]
        M_A = self._variables["M_A"]
        lambda_A = self._variables["lambda_A"]
        C = self._variables["C"]
        alpha = self._variables["alpha"]

        # Recursive emotion update rule
        affective_input = self.negativity_bias(alpha * rpe + (1 - alpha) * r)

        self._variables["M_A"] = (
            M_A
            + (1 - lambda_A)
            * (C * affective_input - M_A)
        )

        # Value learning
        self._variables["V"] = self._variables["V"] + self._variables[
            "eta"] * self._variables["rpe"]


if __name__ == "__main__":
    pass
