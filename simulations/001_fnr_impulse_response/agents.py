import mesa  # agent-based model package
import numpy as np
from typing import Dict, Optional, Union
from scipy.special import softmax
from enum import Enum

RealNumber = Union[float, np.floating]

class IrritabilityAgent(mesa.discrete_space.FixedAgent):
    class Action(Enum):
        NEUTRAL = (0, "theta_N", "p_N")
        FRIENDLY = (1, "theta_F", "p_F")
        AGGRESSIVE = (2, "theta_A", "p_A")

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
        "theta_N_w0",
        "theta_N",       # current tendency for neutral behavior (logits)
        "theta_F_w0",
        "theta_F",       # current tendency for friendly behavior (logits)
        "theta_A_w0",
        "theta_A_w1",
        "theta_A",       # current tendency for aggressive behavior (logits)
        "p_N",           # current probability for friendly behavior
        "p_F",           # current probability for friendly behavior
        "p_A",           # current probability for aggressive behavior
        "a",              # current action chosen
        "r",             # current reward
        "rpe",           # current reward prediction error

        # Parameters
        "lambda_A",
        "C"
    ]

    def __init__(
        self,
        model: mesa.Model,
        V: Optional[RealNumber]=None,
        M_A: Optional[RealNumber]=None,
        theta_N_w0: Optional[RealNumber]=None,
        theta_N: Optional[RealNumber]=None,
        theta_F_w0: Optional[RealNumber]=None,
        theta_F: Optional[RealNumber]=None,
        theta_A_w0: Optional[RealNumber]=None,
        theta_A_w1: Optional[RealNumber]=None,
        theta_A: Optional[RealNumber]=None,
        p_N: Optional[RealNumber]=None,
        p_F: Optional[RealNumber]=None,
        p_A: Optional[RealNumber]=None,
        a: Optional[str]=None,
        r: Optional[RealNumber]=None,
        rpe: Optional[RealNumber]=None,
        lambda_A: Optional[RealNumber]=None,
        C: Optional[RealNumber]=None
    ):
        super().__init__(model)

        # Check invariants

        if theta_N or theta_A or theta_F or p_N or p_A or p_F or a or r or rpe:
            raise ValueError(
                (
                    "Model-implied variables (theta_N, theta_A, theta_F, p_N, "
                    "p_A, p_F, a, r, rpe) must be None in the constructor."
                )
            )

        # C must be in [0,1]
        if C is not None and not (0 <= C <= 1):
            raise ValueError(f"C must be between 0 and 1, got {C}")

        # lambda_A must be in [0,1]
        if lambda_A is not None and not (0 <= lambda_A <= 1):
            raise ValueError(
                f"lambda_A must be between 0 and 1, got {lambda_A}"
            )

        self._variables = {
            "V": V,
            "M_A": M_A,
            "theta_N_w0": theta_N_w0,
            "theta_F_w0": theta_F_w0,
            "theta_A_w0": theta_A_w0,
            "theta_A_w1": theta_A_w1,
            "theta_N": theta_N,
            "theta_A": theta_A,
            "theta_F": theta_F,
            "p_N": p_N,
            "p_F": p_F,
            "p_A": p_A,
            "a": a,
            "r": r,
            "rpe": rpe,
            "lambda_A": lambda_A,
            "C": C
        }

        self.calculate_action_tendencies()

        self._variables["neutral_counter"] = 0
        self._variables["friendly_counter"] = 0
        self._variables["aggressive_counter"] = 0

    def print_id(self):
        print(f"{self.unique_id!s}")

    def print_variables(self):
        print(self._variables)

    def calculate_action_tendencies(self):
        # Calculate logits
        self._variables["theta_N"] = self._variables["theta_N_w0"]

        self._variables["theta_F"] = self._variables["theta_F_w0"]

        self._variables["theta_A"] = self._variables[
            "theta_A_w0"] + self._variables[
                "theta_A_w1"] * self._variables["M_A"]

        # Calculate probabilities
        logits = [self._variables[action.theta_name]
                  for action in sorted(self.Action, key=lambda a: a.value)]

        probs = softmax(logits)

        # Store the "p_F" etc.
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
        if action is self.Action.NEUTRAL:
            self._variables["neutral_counter"] += 1

        elif action is self.Action.FRIENDLY:
            self._variables["friendly_counter"] += 1

        elif action is self.Action.AGGRESSIVE:
            self._variables["aggressive_counter"] += 1

        reward = self.get_reward(action, self._variables)

        rpe = reward - self._variables["V"]

        self._variables["r"] = reward
        self._variables["rpe"] = rpe

        # print(f"{self._action_labels[action]} => {reward} reward, {rpe} RPE")

        return (reward, rpe)

    def get_reward(self, action: "IrritabilityAgent.Action", _variables: Dict):
        # A simple non-reward

        # print(f"calculating reward for step {self.model.steps}")

        if self.model.steps == 1:
            return -1

        else:
            return 0

    def update_emotions_and_action_tendencies(self):
        rpe = self._variables["rpe"]
        M_A = self._variables["M_A"]
        lambda_A = self._variables["lambda_A"]
        C = self._variables["C"]

        # recursive emotion update rule
        self._variables["M_A"] = M_A + (1-lambda_A) * (C*rpe-M_A)

        self.calculate_action_tendencies()
