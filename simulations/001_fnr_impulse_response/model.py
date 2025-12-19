import mesa  # agent-based model package
import numpy as np
import agents
import importlib
importlib.reload(agents)
IrritabilityAgent = agents.IrritabilityAgent


class IrritabilityModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, num_agents=1, seed=None, init_variables=None,
                 init_parameters=None):
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
