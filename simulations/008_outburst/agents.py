import mesa  # agent-based model package
from typing import Union
import math
import numpy as np
from scipy.special import softmax
from enum import Enum

RealNumber = Union[float, np.floating]


class IrritabilityAgent(mesa.discrete_space.FixedAgent):
    class Action(Enum):
        NEUTRAL = (0, "theta_N", "p_N")
        # FRIENDLY = (1, "theta_F", "p_F")
        AGGRESSIVE = (1, "theta_A", "p_A")

        def __new__(cls, number: int, theta_name: str, prob_name: str):
            obj = object.__new__(cls)
            obj._value_ = number
            object.__setattr__(obj, "theta_name", theta_name)
            object.__setattr__(obj, "prob_name", prob_name)
            return obj

    _variable_names = [
        # Variables
        "V",             # estimated value of the current state
        "M_A",           # current anger/frustration
        "M_S",           # current sadness
        "V0",            # initial state value
        "M_A0",          # initial anger/frustration
        "M_S0",          # initial sadness
        "r",             # current reward
        "rpe",           # current reward prediction error
        "v",
        "theta_N_w0",
        "theta_N",       # current tendency for neutral behavior (logits)
        "theta_A_w0",
        "theta_A_w1",
        "theta_A",       # current tendency for aggressive behavior (logits)
        "p_N",           # current probability for neutral behavior
        "p_A",           # current probability for aggressive behavior
        "a",              # current action chosen

        # Parameters
        "lambda_A",
        "C",
        "eta",
        "gamma",
        "alpha",
        "kappa",
        "w_v_A",
        "I",
    ]

    def __init__(
        self,
        model: mesa.Model,
        V0: RealNumber,
        M_A0: RealNumber,
        M_S0: RealNumber,
        C: RealNumber,
        lambda_A: RealNumber,
        eta: RealNumber,
        gamma: RealNumber,
        alpha: RealNumber,
        kappa: RealNumber,
        w_v_A: RealNumber,
        I: RealNumber,
        r1: RealNumber,
        theta_N_w0: RealNumber,
        theta_A_w0: RealNumber,
        theta_A_w1: RealNumber,
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

        # C must be in [0,1]
        if not (0 <= C <= 1):
            raise ValueError(f"C must be between 0 and 1, got {C}")

        # I must be in [0,1]
        if not (0 <= I <= 1):
            raise ValueError(f"I must be between 0 and 1, got {I}")

        self._variables = {
            "V0": V0,
            "M_A0": M_A0,
            "M_S0": M_S0,
            "V": V0,
            "M_A": M_A0,
            "M_S": M_S0,
            "C": C,
            "r": None,
            "rpe": None,
            "lambda_A": lambda_A,
            "eta": eta,
            "gamma": gamma,
            "alpha": alpha,
            "kappa": kappa,
            "w_v_A": w_v_A,
            "v": None,
            "I": I,
            "r1": r1,
            "theta_N_w0": theta_N_w0,
            "theta_A_w0": theta_A_w0,
            "theta_A_w1": theta_A_w1,
            "theta_N": None,
            "theta_A": None,
            "p_N": None,
            "p_A": None,
            "a": None,
            "trial_nr": 1,
        }

        self.calculate_action_tendencies()

    def print_id(self):
        print(f"{self.unique_id!s}")

    def print_variables(self):
        print(self._variables)

    def calculate_action_tendencies(self):
        # Calculate logits
        self._variables["theta_N"] = self._variables["theta_N_w0"]

        self._variables["theta_A"] = self._variables[
            "theta_A_w0"] + self._variables[
                "theta_A_w1"] * self._variables["M_A"]

        # Calculate probabilities
        logits = [self._variables[action.theta_name]
                  for action in sorted(self.Action, key=lambda a: a.value)]

        probs = softmax(logits)

        # Store the probabilities in "p_F" etc.
        for action in sorted(self.Action, key=lambda a: a.value):
            self._variables[action.prob_name] = probs[action.value]

    def choose_action_and_act(self):
        action = self.choose_action()
        self._variables["a"] = action.name
        self.act(action)

    def choose_action(self):
        probs = np.array([
            self._variables[action.prob_name]
            for action in sorted(self.Action, key=lambda a: a.value)
        ])

        # Sample action from a multinomial distribution with the calculated
        # probabilities
        action_choice = self.Action(self.random.choice(len(probs), p=probs))

        return action_choice

    def act(self, action: "IrritabilityAgent.Action") -> tuple:
        self._variables["v"] = self.get_vigor()

        reward = self.get_reward(action)

        # self._variables["r1"]

        rpe = reward + self._variables["gamma"] * self._variables[
            "V"] - self._variables["V"]

        self._variables["r"] = reward
        self._variables["rpe"] = rpe

        self._variables["trial_nr"] += 1

        return (reward, rpe)

    def get_reward(self, action: "IrritabilityAgent.Action"):
        if self.model.steps == 1:
            return self._variables["r1"]

        else:
            return 0

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def get_vigor(self):
        return (1 - self._variables["I"]) * IrritabilityAgent.sigmoid(
            self._variables["w_v_A"] * abs(self._variables["M_A"])
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

    def update_emotions_action_tendencies_and_learn(self):
        self.update_emotions()
        self.calculate_action_tendencies()
        self.learn_state_value()


if __name__ == "__main__":
    pass
