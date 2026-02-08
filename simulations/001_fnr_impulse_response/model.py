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
        theta_N_w0=None,
        theta_N=None,       # current tendency for neutral behavior (logits)
        theta_F_w0=None,
        theta_F=None,       # current tendency for friendly behavior (logits)
        theta_A_w0=None,
        theta_A_w1=None,
        theta_A=None,       # current tendency for aggressive behavior (logits)
        p_N=None,           # current probability for friendly behavior
        p_F=None,           # current probability for friendly behavior
        p_A=None,           # current probability for aggressive behavior
        a=None,             # current action chosen
        r=None,             # current reward
        rpe=None,           # current reward prediction error
        lambda_A=None,
        C=None
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
            theta_N_w0=theta_N_w0,
            theta_N=theta_N,
            theta_F_w0=theta_F_w0,
            theta_F=theta_F,
            theta_A_w0=theta_A_w0,
            theta_A_w1=theta_A_w1,
            theta_A=theta_A,
            p_N=p_N,
            p_F=p_F,
            p_A=p_A,
            a=a,
            r=r,
            rpe=rpe,
            lambda_A=lambda_A,
            C=C
        )

    def step(self):
        self.agents.shuffle_do("choose_action_and_act")

        self.datacollector.collect(self)

        self.agents.shuffle_do("update_emotions_and_action_tendencies")

    @staticmethod
    def agent_portrayal(agent: IrritabilityAgent):
        return mesa.visualization.AgentPortrayalStyle(
            color="tab:orange",
            size=50
        )


if __name__ == "__main__":
    pass
