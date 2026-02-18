import mesa  # agent-based model package
from typing import Union
import math
import numpy as np
from enum import Enum
from scipy.special import softmax

RealNumber = Union[float, np.floating]


class IrritabilityAgent(mesa.discrete_space.FixedAgent):
    class Action(Enum):
        REQUEST_NEUTRALLY = (0, "theta_N", "p_N")
        REQUEST_FRIENDLY = (1, "theta_F", "p_F")
        REQUEST_AGGRESSIVELY = (2, "theta_A", "p_A")

        def __new__(cls, number: int, theta_name: str, prob_name: str):
            obj = object.__new__(cls)
            obj._value_ = number
            object.__setattr__(obj, "theta_name", theta_name)
            object.__setattr__(obj, "prob_name", prob_name)
            return obj

        def __str__(self):
            return self.name

    _variable_names = [
        # Variables
        "V",             # estimated value of the current state
        "M_A",           # current anger/frustration
        "M_S",           # current sadness
        "r",             # current reward
        "rpe",           # current reward prediction error
        "v",
        "theta_N_w0",
        "theta_N",       # current tendency for neutral behavior (logits)
        "theta_F_w0",
        "theta_F",       # current tendency for friendly behavior (logits)
        "theta_A_w0",
        "theta_A_w1",
        "theta_A",       # current tendency for aggressive behavior (logits)
        "p_N",           # current probability for neutral behavior
        "p_F",           # current probability for friendly behavior
        "p_A",           # current probability for aggressive behavior
        "a",             # action

        # Parameters
        "lambda_A",
        "C_start",
        "C_end",
        "lambda_C",
        "midpoint_C",
        "I_start",
        "I_end",
        "lambda_I",
        "midpoint_I",
        "C",
        "I",
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
        V: RealNumber,
        M_A: RealNumber,
        M_S: RealNumber,
        theta_N_w0: RealNumber,
        theta_F_w0: RealNumber,
        theta_A_w0: RealNumber,
        theta_A_w1: RealNumber,
        lambda_A: RealNumber,
        C_start: RealNumber,
        C_end: RealNumber,
        lambda_C: RealNumber,
        midpoint_C: int,
        I_start: RealNumber,
        I_end: RealNumber,
        lambda_I: RealNumber,
        midpoint_I: int,
        eta: RealNumber,
        gamma: RealNumber,
        alpha: RealNumber,
        kappa: RealNumber,
        w_v_A: RealNumber,
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

        # C_start must be in [0,1]
        if not (0 <= C_start <= 1):
            raise ValueError(
                f"C_start must be between 0 and 1, got {C_start}"
            )

        # C_end must be in [0,1]
        if not (0 <= C_end <= 1):
            raise ValueError(
                f"C_end must be between 0 and 1, got {C_end}"
            )

        # lambda_C must be in [0,1]
        if not (0 <= lambda_C <= 1):
            raise ValueError(
                f"lambda_C must be between 0 and 1, got {lambda_C}"
            )

        # midpoint_C must be >= 0
        if not (0 <= midpoint_C):
            raise ValueError(
                f"midpoint_C must be >= 0, got {midpoint_C}"
            )

        # I_start must be in [0,1]
        if not (0 <= I_start <= 1):
            raise ValueError(
                f"I_start must be between 0 and 1, got {I_start}"
            )

        # I_end must be in [0,1]
        if not (0 <= I_end <= 1):
            raise ValueError(
                f"I_end must be between 0 and 1, got {I_end}"
            )

        # lambda_I must be in [0,1]
        if not (0 <= lambda_I <= 1):
            raise ValueError(
                f"lambda_I must be between 0 and 1, got {lambda_I}"
            )

        # midpoint_I must be >= 0
        if not (0 <= midpoint_I):
            raise ValueError(
                f"midpoint_I must be >= 0, got {midpoint_I}"
            )

        self._variables = {
            "V": V,
            "M_A": M_A,
            "M_S": M_S,
            "r": None,
            "rpe": None,
            "lambda_A": lambda_A,
            "C_start": C_start,
            "C_end": C_end,
            "lambda_C": lambda_C,
            "midpoint_C": midpoint_C,
            "C": None,
            "I_start": I_start,
            "I_end": I_end,
            "lambda_I": lambda_I,
            "midpoint_I": midpoint_I,
            "I": None,
            "eta": eta,
            "gamma": gamma,
            "alpha": alpha,
            "kappa": kappa,
            "w_v_A": w_v_A,
            "v": None,
            "theta_N_w0": theta_N_w0,
            "theta_F_w0": theta_F_w0,
            "theta_A_w0": theta_A_w0,
            "theta_A_w1": theta_A_w1,
            "theta_N": None,
            "theta_A": None,
            "theta_F": None,
            "p_N": None,
            "p_F": None,
            "p_A": None,
            "a": None,
            "trial_nr": 1,
            "episode": 0,
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

    def prepare_new_episode(self,
                            V: RealNumber,
                            M_A: RealNumber,
                            M_S: RealNumber,
                            theta_N_w0: RealNumber,
                            theta_F_w0: RealNumber,
                            theta_A_w0: RealNumber,
                            theta_A_w1: RealNumber):

        self._variables["episode"] += 1

        self._variables["V"] = V
        self._variables["M_A"] = M_A
        self._variables["M_S"] = M_S
        self._variables["theta_N_w0"] = theta_N_w0
        self._variables["theta_F_w0"] = theta_F_w0
        self._variables["theta_A_w0"] = theta_A_w0
        self._variables["theta_A_w1"] = theta_A_w1

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

        # Store the probabilities in "p_F" etc.
        for action in sorted(self.Action, key=lambda a: a.value):
            self._variables[action.prob_name] = probs[action.value]

    def choose_action_and_act(self):
        self.calculate_action_tendencies()

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
        v = self.get_vigor()
        self._variables["v"] = v

        if action is self.Action.REQUEST_NEUTRALLY:
            self._variables["neutral_counter"] += 1

        elif action is self.Action.REQUEST_FRIENDLY:
            self._variables["friendly_counter"] += 1

        elif action is self.Action.REQUEST_AGGRESSIVELY:
            self._variables["aggressive_counter"] += 1

        reward = self._environment.act(action, v)

        rpe = reward + self._variables["gamma"] * self._variables[
            "V"] - self._variables["V"]

        self._variables["r"] = reward
        self._variables["rpe"] = rpe

        self._variables["C"] = self.get_controllability()

        self._variables["trial_nr"] += 1

        return (reward, rpe)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def get_vigor(self):
        I = self.get_response_inhibition()

        return (1 - I) * IrritabilityAgent.sigmoid(
            self._variables["w_v_A"] * abs(self._variables["M_A"])
        )

    def negativity_bias(self, affective_input):
        return (
            affective_input
            if affective_input > 0
            else self._variables["kappa"] * affective_input
        )

    def get_controllability(self):
        C_start = self._variables["C_start"]
        C_end = self._variables["C_end"]
        lambda_C = self._variables["lambda_C"]
        midpoint_C = self._variables["midpoint_C"]

        t = self._variables["trial_nr"]
        return C_end + (C_start - C_end) \
            / (1 + math.exp(lambda_C * (t - midpoint_C)))

    def get_response_inhibition(self):
        I_start = self._variables["I_start"]
        I_end = self._variables["I_end"]
        lambda_I = self._variables["lambda_I"]
        midpoint_I = self._variables["midpoint_I"]

        t = self._variables["trial_nr"]
        return I_end + (I_start - I_end) \
            / (1 + math.exp(lambda_I * (t - midpoint_I)))

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
