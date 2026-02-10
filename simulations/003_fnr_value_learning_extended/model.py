import mesa  # agent-based model package
import numpy as np
import agents
import importlib
importlib.reload(agents)
IrritabilityAgent = agents.IrritabilityAgent


class IrritabilityModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(
        self,
        num_agents=1,
        seed=None,
        V=None,             # estimated value of the current state
        M_A=None,           # current anger/frustration
        lambda_A=None,
        C=None,
        eta=None,
        gamma=None,
        alpha=None,
        kappa=None,
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
            lambda_A=lambda_A,
            C=C,
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
