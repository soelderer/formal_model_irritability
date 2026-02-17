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


class IrritabilityModel(mesa.Model):
    """A model with some number of agents."""

    class EnvironmentType(Enum):
        EnvironmentNeutral = auto()

    @dataclass
    class EnvironmentNeutral:
        random: np.random.Generator = field(
            default_factory=np.random.default_rng)

        # This maps actions to probabilities of reward:
        # action: [(probability, reward1), ...]
        transition_table = {
            Action.REQUEST_AGGRESSIVELY: [
                (0.1, +1.0),
                (0.9, 0.0),
            ],

            Action.REQUEST_FRIENDLY: [
                (0.1, +1.0),
                (0.9, 0.0),
            ],

            Action.REQUEST_NEUTRALLY: [
                (0.1, +1.0),
                (0.9, 0.0),
            ],
        }

        def act(
            self,
            action: Action,
            v: RealNumber
        ) -> RealNumber:

            transitions = self.transition_table[action]

            probs = [transition[0] for transition in transitions]

            index = self.random.choice(len(probs), p=probs)
            transition = transitions[index]

            return (transition[1])

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
        match environment_type:
            case self.EnvironmentType.EnvironmentNeutral:
                self.environment = self.EnvironmentNeutral(self.random)
            case _:
                raise NotImplementedError(
                    f"EnvironmentType {environment_type} is not implemented."
                )

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

        # Create agents
        IrritabilityAgent.create_agents(
            model=self,
            n=1,  # number of agents
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
        )

    def step(self):
        self.agents.shuffle_do("act")

        self.datacollector.collect(self)

        self.agents.shuffle_do("update_emotions_and_learn")


if __name__ == "__main__":
    pass
