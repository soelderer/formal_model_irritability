import mesa  # agent-based model package
import numpy as np
import agents
import importlib
importlib.reload(agents)
IrritabilityAgent = agents.IrritabilityAgent
OrthogonalMooreGrid = mesa.discrete_space.OrthogonalMooreGrid


class IrritabilityModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(
        self,
        num_agents=1,
        seed=None,
        V=None,             # estimated value of the current state
        M_A=None,           # current anger/frustration
        r=None,             # current reward
        rpe=None,           # current reward prediction error
        lambda_A=None,
        C=None,
        eta=None,
        gamma=None
    ):
        super().__init__(seed=seed)

        # TODO: check for invariants here (e.g. some parameters must be
        # in [0,1])

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

        # We don't really need space now, so let's just line them up in a grid
        self.grid = OrthogonalMooreGrid(
            (num_agents, 1),
            torus=True,
            capacity=num_agents
        )

        # Create agents
        IrritabilityAgent.create_agents(
            model=self,
            n=1,  # number of agents
            V=V,
            M_A=M_A,
            r=r,
            rpe=rpe,
            lambda_A=lambda_A,
            C=C,
            eta=eta,
            gamma=gamma
        )

    def step(self):
        self.agents.shuffle_do("act")

        self.datacollector.collect(self)

        self.agents.shuffle_do("update_emotions_and_learn")

    @staticmethod
    def agent_portrayal(agent: IrritabilityAgent):
        return mesa.visualization.AgentPortrayalStyle(
            color="tab:orange",
            size=50
        )


if __name__ == "__main__":
    pass
