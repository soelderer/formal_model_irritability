import mesa  # agent-based model package
from typing import Union
import numpy as np

RealNumber = Union[float, np.floating]


class IrritabilityAgent(mesa.discrete_space.FixedAgent):
    _variable_names = [
        # Variables
        "V",             # estimated value of the current state
        "M_A",           # current anger/frustration
        "r",             # current reward
        "rpe",           # current reward prediction error

        # Parameters
        "lambda_A_pos",
        "lambda_A_neg",
        "C",
        "eta",
        "gamma",
        "alpha",
        "kappa"
    ]

    def __init__(
        self,
        model: mesa.Model,
        V: RealNumber,
        M_A: RealNumber,
        lambda_A_pos: RealNumber,
        lambda_A_neg: RealNumber,
        C: RealNumber,
        eta: RealNumber,
        gamma: RealNumber,
        alpha: RealNumber,
        kappa: RealNumber
    ):
        super().__init__(model)

        # C must be in [0,1]
        if not (0 <= C <= 1):
            raise ValueError(f"C must be between 0 and 1, got {C}")

        # lambda_A_pos must be in [0,1]
        if not (0 <= lambda_A_pos <= 1):
            raise ValueError(
                f"lambda_A_pos must be between 0 and 1, got {lambda_A_pos}"
            )

        # lambda_A_neg must be in [0,1]
        if not (0 <= lambda_A_neg <= 1):
            raise ValueError(
                f"lambda_A_neg must be between 0 and 1, got {lambda_A_neg}"
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

        self._variables = {
            "V": V,
            "M_A": M_A,
            "r": None,
            "rpe": None,
            "lambda_A_pos": lambda_A_pos,
            "lambda_A_neg": lambda_A_neg,
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

        if self._variables["trial_nr"] % 100 == 0 and (
                self._variables["block_nr"] == 1 or
                self._variables["block_nr"] == 2):
            self._variables["block_nr"] += 1

        return (reward, rpe)

    def get_reward(self):
        if self._variables["block_nr"] == 1 or \
                self._variables["block_nr"] == 3:
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

    def update_emotions(self):
        r = self._variables["r"]
        rpe = self._variables["rpe"]
        M_A = self._variables["M_A"]
        lambda_A_pos = self._variables["lambda_A_pos"]
        lambda_A_neg = self._variables["lambda_A_neg"]
        C = self._variables["C"]
        alpha = self._variables["alpha"]

        # Recursive emotion update rule
        affective_input = self.negativity_bias(alpha * rpe + (1 - alpha) * r)

        if M_A >= 0:
            lambda_A = lambda_A_pos
        else:
            lambda_A = lambda_A_neg

        self._variables["M_A"] = (
            M_A
            + (1 - lambda_A)
            * (C * affective_input - M_A)
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
