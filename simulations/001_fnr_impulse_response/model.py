import mesa  # agent-based model package
import numpy as np
import seaborn as sns
from typing import Dict
from scipy.special import softmax


class IrritabilityAgent(mesa.Agent):
    _variable_names = [
        "V",             # estimated value of the current state
        "M_A",           # current anger/frustration
        "theta_F_w0",
        "theta_F_w1",
        "theta_F",       # current tendency for friendly behavior (logits)
        "theta_A_w0",
        "theta_A_w1",
        "theta_A",       # current tendency for aggressive behavior (logits)
        "p_F",           # current probability for friendly behavior
        "p_A",           # current probability for aggressive behavior
        "a",              # current action chosen
        "r",             # current reward
        "rpe",           # current reward prediction error
    ]

    _parameter_names = [
        "lambda_A",
        "C"
    ]

    def __init__(self, model, init_variables: Dict, init_parameters: Dict):
        super().__init__(model)

        # TODO: check if init_variable.keys() match _variable_names

        # TODO: check if init_parameters.keys() match _parameter_names

        # TODO: check for invariants here (e.g. some parameters must be
        # in [0,1])

        # raise ValueError()

        self._variables = init_variables
        self._parameters = init_parameters

        # TODO: Maybe refactor into enum?
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

    def choose_action_and_act(self):
        action = self.choose_action()
        self._variables["a"] = self._action_labels[action]
        self.act(action)

    def choose_action(self):
        logits = [self._variables["theta_F"], self._variables["theta_A"]]
        probs = softmax(logits)

        # TODO: good idea to use this RNG? we should be consistent throughout
        # the project
        action = np.random.choice(len(probs), p=probs)

        # TODO: maybe refactor with enum would be more explicit about indices
        self._variables["p_F"] = probs[0]
        self._variables["p_A"] = probs[1]

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

        self._variables["r"] = reward
        self._variables["rpe"] = rpe

        # print(f"{self._action_labels[action]} => {reward} reward, {rpe} RPE")

        return (reward, rpe)

    def get_reward(self, action, _variables: Dict):
        # a simple non-reward

        # print(f"calculating reward for step {self.model.steps}")

        if self.model.steps == 1:
            return -1

        else:
            return 0

    def update_emotions_and_action_tendencies(self):
        rpe = self._variables["rpe"]
        M_A = self._variables["M_A"]
        lambda_A = self._parameters["lambda_A"]

        # recursive emotion update rule
        self._variables["M_A"] = M_A + (1-lambda_A) * (rpe-M_A)

        self.calculate_action_tendencies()


class IrritabilityModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, num_agents=1, seed=None, init_variables=None,
                 init_parameters=None):
        super().__init__(seed=seed)

        # TODO: check for invariants here (e.g. some parameters must be
        # in [0,1])

        self.num_agents_ = num_agents

        # Instantiate DataCollector
        self.datacollector = mesa.DataCollector(
            model_reporters=None, agent_reporters={
                **{
                    name: (lambda agent, key=name: agent._variables[key])
                    for name in IrritabilityAgent._variable_names
                },
                **{
                    name: (lambda agent, key=name: agent._parameters[key])
                    for name in IrritabilityAgent._parameter_names
                },
            }
        )

        # Create agents
        IrritabilityAgent.create_agents(model=self,
                                        n=1,  # number of agents
                                        init_variables=init_variables,
                                        init_parameters=init_parameters)

    def step(self):
        self.agents.shuffle_do("choose_action_and_act")

        self.datacollector.collect(self)

        self.agents.shuffle_do("update_emotions_and_action_tendencies")


if __name__ == "__main__":
    pass
