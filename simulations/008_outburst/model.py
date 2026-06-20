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
        V0: RealNumber,             # estimated value of the current state
        M_A0: RealNumber,           # current anger/frustration
        M_S0: RealNumber,           # current sadness
        C: RealNumber,
        lambda_A: RealNumber,
        eta: RealNumber,
        gamma: RealNumber,
        alpha: RealNumber,
        kappa: RealNumber,
        w_v_A: RealNumber,
        I: RealNumber,
        r1: RealNumber,
        theta_N_w0: RealNumber,
        theta_A_w0: RealNumber,
        theta_A_w1: RealNumber,
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
            V0=V0,
            M_A0=M_A0,
            M_S0=M_S0,
            C=C,
            lambda_A=lambda_A,
            eta=eta,
            gamma=gamma,
            alpha=alpha,
            kappa=kappa,
            w_v_A=w_v_A,
            I=I,
            r1=r1,
            theta_N_w0=theta_N_w0,
            theta_A_w0=theta_A_w0,
            theta_A_w1=theta_A_w1,
        )

    def step(self):
        self.agents.shuffle_do("choose_action_and_act")

        self.datacollector.collect(self)

        self.agents.shuffle_do("update_emotions_action_tendencies_and_learn")


if __name__ == "__main__":
    pass
