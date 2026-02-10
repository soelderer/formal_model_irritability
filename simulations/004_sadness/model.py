import mesa  # agent-based model package
import numpy as np
from typing import Union
import agents
import importlib
importlib.reload(agents)
IrritabilityAgent = agents.IrritabilityAgent

RealNumber = Union[float, np.floating]


class IrritabilityModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(
        self,
        seed: int,
        V: RealNumber,             # estimated value of the current state
        M_A: RealNumber,           # current anger/frustration
        M_S: RealNumber,           # current sadness
        lambda_A: RealNumber,
        lambda_C: RealNumber,
        midpoint: int,
        eta: RealNumber,
        gamma: RealNumber,
        alpha: RealNumber,
        kappa: RealNumber,
        num_agents=1,
    ):
        super().__init__()

        self.num_agents_ = num_agents

        # Override default RNG to numpy (supports multinomial distribution)
        self.random = np.random.default_rng(seed)

        # Instantiate DataCollector
        self.datacollector = mesa.DataCollector(
            model_reporters=None, agent_reporters={
                **{
                    name: (lambda agent, key=name: agent._variables[key])
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
            r=None,
            rpe=None,
            lambda_A=lambda_A,
            lambda_C=lambda_C,
            midpoint=midpoint,
            C=None,
            eta=eta,
            gamma=gamma,
            alpha=alpha,
            kappa=kappa
        )

    def step(self):
        self.agents.shuffle_do("act")

        self.datacollector.collect(self)

        self.agents.shuffle_do("update_emotions_and_learn")


if __name__ == "__main__":
    pass
