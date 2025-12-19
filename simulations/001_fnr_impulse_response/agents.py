import mesa  # agent-based model package
from typing import Dict
from scipy.special import softmax
from enum import Enum


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
        "theta_N_w1",
        "theta_N",       # current tendency for neutral behavior (logits)
        "theta_F_w0",
        "theta_F_w1",
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
        model,
        V=None,
        M_A=None,
        theta_N_w0=None,
        theta_N_w1=None,
        theta_N=None,
        theta_F_w0=None,
        theta_F_w1=None,
        theta_F=None,
        theta_A_w0=None,
        theta_A_w1=None,
        theta_A=None,
        p_N=None,
        p_F=None,
        p_A=None,
        a=None,
        r=None,
        rpe=None,
        lambda_A=None,
        C=None
    ):
        super().__init__(model)

        # TODO: check if init_variable.keys() match _variable_names

        # TODO: check for invariants here (e.g. some parameters must be
        # in [0,1])

        # raise ValueError()

        self._variables = {
            "V": V,
            "M_A": M_A,
            "theta_N_w0": theta_N_w0,
            "theta_N_w1": theta_N_w1,
            "theta_N": theta_N,
            "theta_F_w0": theta_F_w0,
            "theta_F_w1": theta_F_w1,
            "theta_F": theta_F,
            "theta_A_w0": theta_A_w0,
            "theta_A_w1": theta_A_w1,
            "theta_A": theta_A,
            "p_N": p_N,
            "p_F": p_F,
            "p_A": p_A,
            "a": a,
            "r": r,
            "rpe": rpe,
            "lambda_A": lambda_A,
            "C": C
        }

        self._variables["neutral_counter"] = 0
        self._variables["friendly_counter"] = 0
        self._variables["aggressive_counter"] = 0

    def print_id(self):
        print(f"{self.unique_id!s}")

    def print_variables(self):
        print(self._variables)

    def calculate_action_tendencies(self):
        self._variables["theta_N"] = self._variables["theta_N_w0"]

        self._variables["theta_F"] = self._variables["theta_F_w0"]

        self._variables["theta_A"] = self._variables[
            "theta_A_w0"] + self._variables[
                "theta_A_w1"] * self._variables["M_A"]

    def choose_action_and_act(self):
        action = self.choose_action()
        self._variables["a"] = action.name
        self.act(action)

    def choose_action(self):
        # Access the "theta_F" etc.
        logits = [self._variables[action.theta_name]
                  for action in sorted(self.Action, key=lambda a: a.value)]

        probs = softmax(logits)

        # Store the "p_F" etc.
        for action in sorted(self.Action, key=lambda a: a.value):
            self._variables[action.prob_name] = probs[action.value]

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
