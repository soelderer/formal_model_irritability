import mesa  # agent-based model package
from typing import Union
from scipy.special import softmax
import math
import numpy as np
from enum import Enum, auto

RealNumber = Union[float, np.floating]


class IrritabilityAgent(mesa.discrete_space.FixedAgent):
    class Action(Enum):
        REQUEST_NEUTRALLY = (0, "theta_N", "p_N")
        REQUEST_FRIENDLY = (1, "theta_F", "p_F")
        REQUEST_AGGRESSIVELY = (2, "theta_A", "p_A")
        DO_STH_ELSE = (3, "theta_E", "p_E")

        def __new__(cls, number: int, theta_name: str, prob_name: str):
            obj = object.__new__(cls)
            obj._value_ = number
            object.__setattr__(obj, "theta_name", theta_name)
            object.__setattr__(obj, "prob_name", prob_name)
            return obj

    class State(Enum):
        GOAL_NOT_REQUESTED = auto()
        GOAL_DENIED = auto()
        GOAL_ABANDONED = auto()
        GOAL_GRANTED = auto()

    _variable_names = [
        # Variables

        "S",             # current state

        # State values
        "V_NR",          # Value of state GOAL_NOT_REQUESTED
        "V_DE",          # Value of state GOAL_DENIED
        "V_AB",          # Value of state GOAL_ABANDONED
        "V_GR",          # Value of state GOAL_GRANTED
        "M_A",           # current anger/frustration
        "M_S",           # current sadness
        "r",             # current reward
        "rpe",           # current reward prediction error

        # Action tendencies
        "v",             # response vigor
        "theta_N_w0",
        "theta_N",       # current tendency for neutral behavior (logits)
        "theta_F_w0",
        "theta_F",       # current tendency for friendly behavior (logits)
        "theta_A_w0",
        "theta_A_w1",
        "theta_A",       # current tendency for aggressive behavior (logits)
        "theta_E_w0",
        "theta_E",       # current tendency for doing something else (logits)
        "p_N",           # current probability for neutral behavior
        "p_F",           # current probability for friendly behavior
        "p_A",           # current probability for aggressive behavior
        "a",             # action

        # Parameters
        "lambda_A",
        "C",
        "eta",
        "gamma",
        "alpha",
        "kappa",
        "w_v_A"
    ]

    def __init__(
        self,
        model: mesa.Model,
        environment,
        S: State,
        V_NR: RealNumber,
        V_DE: RealNumber,
        V_AB: RealNumber,
        V_GR: RealNumber,
        M_A: RealNumber,
        M_S: RealNumber,
        theta_N_w0: RealNumber,
        theta_F_w0: RealNumber,
        theta_A_w0: RealNumber,
        theta_A_w1: RealNumber,
        theta_E_w0: RealNumber,
        lambda_A: RealNumber,
        eta: RealNumber,
        gamma: RealNumber,
        alpha: RealNumber,
        kappa: RealNumber,
        w_v_A: RealNumber,
        C: RealNumber,
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

        # C must be in [0,1]
        if not (0 <= C <= 1):
            raise ValueError(f"C must be between 0 and 1, got {C}")

        # kappa must be in R+
        if not (kappa >= 1):
            raise ValueError(f"kappa must be >=1, got {kappa}")

        self._variables = {
            "S": S,
            "V_NR": V_NR,
            "V_DE": V_DE,
            "V_AB": V_AB,
            "V_GR": V_GR,
            "M_A": M_A,
            "M_S": M_S,
            "theta_N_w0": theta_N_w0,
            "theta_F_w0": theta_F_w0,
            "theta_A_w0": theta_A_w0,
            "theta_A_w1": theta_A_w1,
            "theta_E_w0": theta_E_w0,
            "theta_N": None,
            "theta_A": None,
            "theta_F": None,
            "p_N": None,
            "p_F": None,
            "p_A": None,
            "a": None,
            "r": None,
            "rpe": None,
            "a": None,
            "lambda_A": lambda_A,
            "C": C,
            "eta": eta,
            "gamma": gamma,
            "alpha": alpha,
            "kappa": kappa,
            "w_v_A": w_v_A,
            "v": None,
        }

        self._environment = environment

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

        self._variables["theta_E"] = self._variables["theta_E_w0"]

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

        print(f"State: {self._variables["S"]}")
        print(f"Chose action: {action.name}")

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
        v = self.get_vigor()
        self._variables["v"] = v

        if action is self.Action.REQUEST_NEUTRALLY:
            self._variables["neutral_counter"] += 1

        elif action is self.Action.REQUEST_FRIENDLY:
            self._variables["friendly_counter"] += 1

        elif action is self.Action.REQUEST_AGGRESSIVELY:
            self._variables["aggressive_counter"] += 1

        print(f"act {action.name}: get_state_value()")
        V_old = self.get_state_value()

        self._S_new, reward = self._environment.act(
            self._variables["S"],
            action,
            v
        )

        print(f"self._S_new: {self._S_new}")
        print(f"reward: {reward}")

        rpe = reward + self._variables["gamma"] * V_old - V_old

        self._variables["r"] = reward
        self._variables["rpe"] = rpe

        return (reward, rpe)

    def get_state_value(self) -> RealNumber:
        print(f"Getting state value of {self._variables["S"]}...")

        match self._variables["S"]:
            case self.State.GOAL_NOT_REQUESTED:
                return self._variables["V_NR"]

            case self.State.GOAL_DENIED:
                return self._variables["V_DE"]

            case _:
                raise ValueError(
                    ("Tried to get the value of a terminal state: ",
                     f"{self._variables["S"]}."
                     )
                )

    def set_state_value(self, V: RealNumber):
        match self._variables["S"]:
            case self.State.GOAL_NOT_REQUESTED:
                self._variables["V_NR"] = V

            case self.State.GOAL_DENIED:
                self._variables["V_DE"] = V

            case _:
                raise ValueError(
                    ("Tried to set the value of a terminal state: ",
                     f"{self._variables["S"]}."
                     )
                )

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def get_vigor(self):
        return IrritabilityAgent.sigmoid(
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
        print("learning state value...")

        V = self.get_state_value()

        V += self._variables["eta"] * self._variables["rpe"]

        self.set_state_value(V)

    def get_state(self):
        return self._variables["S"]

    def transit_state(self):
        self._variables["S"] = self._S_new
        return self._S_new

    def update_emotions_and_learn(self):
        self.update_emotions()
        self.learn_state_value()


if __name__ == "__main__":
    pass
