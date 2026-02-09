import mesa  # agent-based model package
import numpy as np
import agents
import importlib
importlib.reload(agents)
from typing import Dict, Optional, Union
IrritabilityAgent = agents.IrritabilityAgent

RealNumber = Union[float, np.floating]

class IrritabilityModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(
        self,
        seed,
        V: RealNumber,             # estimated value of the current state
        M_A: RealNumber,           # current anger/frustration
        theta_N_w0: RealNumber,
        theta_F_w0: RealNumber,
        theta_A_w0: RealNumber,
        theta_A_w1: RealNumber,
        lambda_A: RealNumber,
        C: RealNumber,
        num_agents=1,
    ):
        super().__init__(seed=seed)

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
            theta_N_w0=theta_N_w0,
            theta_F_w0=theta_F_w0,
            theta_A_w0=theta_A_w0,
            theta_A_w1=theta_A_w1,
            lambda_A=lambda_A,
            C=C
        )

    def step(self):
        self.agents.shuffle_do("choose_action_and_act")

        self.datacollector.collect(self)

        self.agents.shuffle_do("update_emotions_and_action_tendencies")

if __name__ == "__main__":
    pass
