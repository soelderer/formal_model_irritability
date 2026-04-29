import mesa  # agent-based model package
from typing import Union
import math
import numpy as np

RealNumber = Union[float, np.floating]


class IrritabilityAgent(mesa.discrete_space.FixedAgent):
    _variable_names = [
        # Variables
        "V",             # estimated value of the current state
        "M_A",           # current anger/frustration
        "M_S",           # current sadness
        "r",             # current reward
        "rpe",           # current reward prediction error
        "v",

        # Parameters
        "lambda_A",
        "lambda_C",
        "midpoint",
        "C",
        "eta",
        "gamma",
        "alpha",
        "kappa",
        "w_v_A",
        "I"
    ]

    def __init__(
        self,
        model: mesa.Model,
        V: RealNumber,
        M_A: RealNumber,
        M_S: RealNumber,
        lambda_A: RealNumber,
        lambda_C: RealNumber,
        midpoint: int,
        eta: RealNumber,
        gamma: RealNumber,
        alpha: RealNumber,
        kappa: RealNumber,
        w_v_A: RealNumber,
        I: RealNumber
    ):
        super().__init__(model)

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

        # alpha must be in [0,1]
        if not (0 <= alpha <= 1):
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

        # kappa must be in R+
        if not (kappa >= 1):
            raise ValueError(f"kappa must be >=1, got {kappa}")

        # lambda_C must be in [0,1]
        if not (0 <= lambda_C <= 1):
            raise ValueError(
                f"lambda_C must be between 0 and 1, got {lambda_C}"
            )

        # midpoint must be in [100, 200]
        if not (100 <= midpoint <= 200):
            raise ValueError(
                f"midpoint must be between 100 and 200, got {midpoint}"
            )

        # I must be in [0,1]
        if not (0 <= I <= 1):
            raise ValueError(f"I must be between 0 and 1, got {I}")

        self._variables = {
            "V": V,
            "M_A": M_A,
            "M_S": M_S,
            "r": None,
            "rpe": None,
            "lambda_A": lambda_A,
            "lambda_C": lambda_C,
            "midpoint": midpoint,
            "C": None,
            "eta": eta,
            "gamma": gamma,
            "alpha": alpha,
            "kappa": kappa,
            "w_v_A": w_v_A,
            "v": None,
            "I": I,
            "trial_nr": 1,
            "block_nr": 1
        }

    def print_id(self):
        print(f"{self.unique_id!s}")

    def print_variables(self):
        print(self._variables)

    def act(self) -> tuple:
        self._variables["v"] = self.get_vigor()

        reward = self.get_reward()

        rpe = reward + self._variables["gamma"] * self._variables[
            "V"] - self._variables["V"]

        self._variables["r"] = reward
        self._variables["rpe"] = rpe

        self._variables["C"] = self.get_controllability()

        self._variables["trial_nr"] += 1

        if self._variables["trial_nr"] == 100 and self._variables[
            "block_nr"
        ] == 1:
            self._variables["block_nr"] += 1

        return (reward, rpe)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def get_vigor(self):
        return (1 - self._variables["I"]) * IrritabilityAgent.sigmoid(
            self._variables["w_v_A"] * abs(self._variables["M_A"])
        )

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

    def get_controllability(self):
        lambda_C = self._variables["lambda_C"]
        midpoint = self._variables["midpoint"]

        if self._variables["block_nr"] == 1:
            return 1.0
        elif self._variables["block_nr"] == 2:
            t = self._variables["trial_nr"]
            return 1 / (1 + math.exp(lambda_C * (t - midpoint)))

    def update_emotions(self):
        r = self._variables["r"]
        rpe = self._variables["rpe"]
        M_A = self._variables["M_A"]
        M_S = self._variables["M_S"]
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

        self._variables["M_S"] = (
            M_S
            + (1 - lambda_A)
            * ((1 - C) * affective_input - M_S)
        )

    def learn_state_value(self):
        # Value learning
        self._variables["V"] = self._variables["V"] + self._variables[
            "eta"] * self._variables["rpe"]

    def update_emotions_and_learn(self):
        self.update_emotions()
        self.learn_state_value()


if __name__ == "__main__":
    pass
