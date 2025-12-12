import mesa  # agent-based model package
import numpy as np
import seaborn as sns
from typing import Dict
from scipy.special import softmax


class IrritabilityAgent(mesa.Agent):
    def __init__(self, model, init_variables: Dict, init_parameters: Dict):
        super().__init__(model)

        # TODO: check for invariants here (e.g. some parameters must be
        # in [0,1])

        self._variables = init_variables
        self._parameters = init_parameters

        self._action_labels = {
            0: "friendly",
            1: "aggressive"
        }

        self._variables["aggressive_counter"] = 0
        self._variables["friendly_counter"] = 0

    def print_id(self):
        print(f"{self.unique_id!s}")

    def print_variables(self):
        print(self._variables)

    def print_parameters(self):
        print(self._parameters)

    def calculate_action_tendencies(self):
        self._variables["theta_F"] = self._variables["theta_F_w0"]

        self._variables["theta_A"] = self._variables[
            "theta_A_w0"] + self._variables[
                "theta_A_w1"] * self._variables["M_A"]

    def step(self):
        self.calculate_action_tendencies()
        action = self.choose_action()

        reward, rpe = self.act(action)
        self.update_variables(reward, rpe)

        self.print_variables()

    def choose_action(self):
        logits = [self._variables["theta_F"], self._variables["theta_A"]]
        probs = softmax(logits)

        # TODO: good idea to use this RNG? we should be consistent throughout
        # the project
        action = np.random.choice(len(probs), p=probs)

        action_probs = [(self._action_labels[action], f"{probs[action]:.3f}")
                        for action in self._action_labels.keys()]

        print(action_probs)

        return action

    def act(self, action: int) -> tuple:
        if action not in self._action_labels.keys():
            raise ValueError(f"Unknown action {action}")

        if self._action_labels[action] == "aggressive":
            self._variables["aggressive_counter"] += 1

        elif self._action_labels[action] == "friendly":
            self._variables["friendly_counter"] += 1

        reward = self.get_reward(action, self._variables)

        rpe = reward - self._variables["V"]

        print(f"{self._action_labels[action]} => {reward} reward, {rpe} RPE")

        return (reward, rpe)

    def get_reward(self, action, _variables: Dict):
        # a simple non-reward

        print(f"calculating reward for step {self.model.steps}")

        if self.model.steps == 1:
            return -1

        else:
            return 0

    def update_variables(self, reward, rpe):
        # recursive emotion update rule
        self._variables["M_A"] = self._variables["M_A"] + (1-self._parameters[
            "lambda_A"]) * (rpe - self._variables["M_A"])


class IrritabilityModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, num_agents=1, seed=None, init_variables=None,
                 init_parameters=None):
        super().__init__(seed=seed)

        # TODO: check for invariants here (e.g. some parameters must be
        # in [0,1])

        self.num_agents_ = num_agents

        # Create agents
        IrritabilityAgent.create_agents(model=self,
                                        n=1,  # number of agents
                                        init_variables=init_variables,
                                        init_parameters=init_parameters)

    def step(self):
        self.agents.shuffle_do("step")


if __name__ == "__main__":
    pass
