from dataclasses import dataclass, field
from enum import Enum, auto
import mesa  # agent-based model package
import numpy as np
from typing import Union
import agents
import importlib
importlib.reload(agents)
IrritabilityAgent = agents.IrritabilityAgent
Action = agents.IrritabilityAgent.Action

RealNumber = Union[float, np.floating]


class EnvironmentType(Enum):
    EnvironmentNeutral = auto()
    EnvironmentAversive = auto()
    EnvironmentAppetitive = auto()


@dataclass
class Environment:
    _environment_type: EnvironmentType

    _random: np.random.Generator

    # This maps actions to probabilities of reward:
    # action: [(probability, reward1), ...]
    _transition_table = {
        EnvironmentType.EnvironmentNeutral: {
            Action.REQUEST_AGGRESSIVELY: [
                (0.2, +2.0),
                (0.7, 0.0),
                (0.1, -2.0),
            ],

            Action.REQUEST_FRIENDLY: [
                (0.2, +2.0),
                (0.7, 0.0),
                (0.1, -2.0),
            ],

            Action.REQUEST_NEUTRALLY: [
                (0.2, +2.0),
                (0.7, 0.0),
                (0.1, -2.0),
            ],
        },

        EnvironmentType.EnvironmentAversive: {
            Action.REQUEST_AGGRESSIVELY: [
                (0.1, +2.0),
                (0.2, 0.0),
                (0.7, -2.0),
            ],

            Action.REQUEST_FRIENDLY: [
                (0.1, +2.0),
                (0.2, 0.0),
                (0.7, -2.0),
            ],

            Action.REQUEST_NEUTRALLY: [
                (0.1, +2.0),
                (0.2, 0.0),
                (0.7, -2.0),
            ],
        },

        EnvironmentType.EnvironmentAppetitive: {
            Action.REQUEST_AGGRESSIVELY: [
                (0.7, +2.0),
                (0.2, 0.0),
                (0.1, -2.0),
            ],

            Action.REQUEST_FRIENDLY: [
                (0.7, +2.0),
                (0.2, 0.0),
                (0.1, -2.0),
            ],

            Action.REQUEST_NEUTRALLY: [
                (0.7, +2.0),
                (0.2, 0.0),
                (0.1, -2.0),
            ],
        }
    }

    def act(
        self,
        action: Action,
        v: RealNumber
    ) -> RealNumber:

        transitions = self._transition_table[self._environment_type][action]

        probs = [transition[0] for transition in transitions]

        index = self._random.choice(len(probs), p=probs)
        transition = transitions[index]

        return (transition[1])


class IrritabilityModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(
        self,
        seed: int,
        environment_type: EnvironmentType,
        V: RealNumber,             # estimated value of the current state
        M_A: RealNumber,           # current anger/frustration
        M_S: RealNumber,           # current sadness
        lambda_A: RealNumber,
        theta_N_w0: RealNumber,
        theta_F_w0: RealNumber,
        theta_A_w0: RealNumber,
        theta_A_w1: RealNumber,
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
        num_agents=1,
    ):
        super().__init__()

        self.num_agents_ = num_agents

        # Override default RNG to numpy (supports multinomial distribution)
        self.random = np.random.default_rng(seed)

        # Instantiate Environment
        self.environment = Environment(environment_type, self.random)

        # Instantiate DataCollector
        self.datacollector = mesa.DataCollector(
            model_reporters=None, agent_reporters={
                **{
                    name: (lambda agent, key=name:
                           agent._variables[key].name
                           if isinstance(agent._variables[key], Enum)
                           else agent._variables[key])
                    for name in IrritabilityAgent._variable_names
                }
            }
        )

        # Save initial state values and emotions for preparing later episodes
        self._V_init = V
        self._M_A_init = M_A
        self._M_S_init = M_S
        self._theta_N_w0_init = theta_N_w0
        self._theta_F_w0_init = theta_F_w0
        self._theta_A_w0_init = theta_A_w0
        self._theta_A_w1_init = theta_A_w1

        # Create agents
        IrritabilityAgent.create_agents(
            model=self,
            n=1,  # number of agents
            environment=self.environment,
            V=V,
            M_A=M_A,
            M_S=M_S,
            lambda_A=lambda_A,
            C_start=C_start,
            C_end=C_end,
            lambda_C=lambda_C,
            midpoint_C=midpoint_C,
            I_start=I_start,
            I_end=I_end,
            lambda_I=lambda_I,
            midpoint_I=midpoint_I,
            eta=eta,
            gamma=gamma,
            alpha=alpha,
            kappa=kappa,
            w_v_A=w_v_A,
            theta_N_w0=theta_N_w0,
            theta_F_w0=theta_F_w0,
            theta_A_w0=theta_A_w0,
            theta_A_w1=theta_A_w1,
        )

    def step(self):
        new_episode = self.steps % 2 == 1

        if new_episode:
            self.agents.shuffle_do(
                "prepare_new_episode",
                V=self._V_init,
                M_A=self._M_A_init,
                M_S=self._M_S_init,
                theta_N_w0=self._theta_N_w0_init,
                theta_F_w0=self._theta_F_w0_init,
                theta_A_w0=self._theta_A_w0_init,
                theta_A_w1=self._theta_A_w1_init,
            )

            self.agents.shuffle_do("choose_action_and_act")

            self.datacollector.collect(self)

            self.agents.shuffle_do("update_emotions_and_learn")

        else:
            self.agents.shuffle_do("choose_action_and_act")

            self.datacollector.collect(self)

            # No need to update emotions because episode is over


if __name__ == "__main__":
    pass
