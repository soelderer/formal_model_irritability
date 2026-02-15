from dataclasses import dataclass, field
import mesa  # agent-based model package
import numpy as np
from typing import Union
from enum import Enum, auto
import agents
IrritabilityAgent = agents.IrritabilityAgent
State = agents.IrritabilityAgent.State
Action = agents.IrritabilityAgent.Action

RealNumber = Union[float, np.floating]


class IrritabilityModel(mesa.Model):
    """A model with some number of agents."""

    # TODO: actually work through the nubmers and maybe have different classes
    # EnvironmentBenign, EnvironmentNeutral, EnvironmentInconsistent...

    class EnvironmentType(Enum):
        EnvironmentNeutral = auto()

    @dataclass
    class EnvironmentNeutral:
        random: np.random.Generator = field(
            default_factory=np.random.default_rng)

        # This maps old-state-action pairs to probabilities of new-state-reward
        # pairs: (S_old, action) -> (p, S_new, r)
        transition_table = {
            (State.GOAL_NOT_REQUESTED, Action.REQUEST_AGGRESSIVELY): [
                (0.2, State.GOAL_GRANTED, +1.0),
                (0.8, State.GOAL_DENIED, 0.0),
            ],

            (State.GOAL_NOT_REQUESTED, Action.REQUEST_FRIENDLY): [
                (0.2, State.GOAL_GRANTED, +1.0),
                (0.8, State.GOAL_DENIED, 0.0),
            ],

            (State.GOAL_NOT_REQUESTED, Action.REQUEST_NEUTRALLY): [
                (0.2, State.GOAL_GRANTED, +1.0),
                (0.8, State.GOAL_DENIED, 0.0),
            ],

            (State.GOAL_NOT_REQUESTED, Action.DO_STH_ELSE): [
                (0.0, State.GOAL_ABANDONED, +1.0),
                (1.0, State.GOAL_ABANDONED, 0.0),
            ],

            (State.GOAL_DENIED, Action.REQUEST_AGGRESSIVELY): [
                (0.2, State.GOAL_GRANTED, +1.0),
                (0.8, State.GOAL_DENIED, 0.0),
            ],

            (State.GOAL_DENIED, Action.REQUEST_FRIENDLY): [
                (0.2, State.GOAL_GRANTED, +1.0),
                (0.8, State.GOAL_DENIED, 0.0),
            ],

            (State.GOAL_DENIED, Action.REQUEST_NEUTRALLY): [
                (0.2, State.GOAL_GRANTED, +1.0),
                (0.8, State.GOAL_DENIED, 0.0),
            ],

            (State.GOAL_DENIED, Action.DO_STH_ELSE): [
                (0.0, State.GOAL_ABANDONED, +1.0),
                (1.0, State.GOAL_ABANDONED, 0.0),
            ],
        }

        def act(
            self,
            S_old: State,
            action: Action,
            v: RealNumber
        ) -> tuple[State, RealNumber]:

            transitions = self.transition_table[(S_old), (action)]

            probs = [transition[0] for transition in transitions]

            index = self.random.choice(len(probs), p=probs)
            transition = transitions[index]

            return (transition[1], transition[2])

    def __init__(
        self,
        seed: int,
        environment_type: EnvironmentType,
        S: State,
        V_NR: RealNumber,
        V_DE: RealNumber,
        V_AB: RealNumber,
        V_GR: RealNumber,
        M_A: RealNumber,           # current anger/frustration
        M_S: RealNumber,           # current sadness
        theta_N_w0: RealNumber,
        theta_F_w0: RealNumber,
        theta_A_w0: RealNumber,
        theta_A_w1: RealNumber,
        theta_E_w0: RealNumber,
        lambda_A: RealNumber,
        eta: RealNumber,
        gamma: RealNumber,
        alpha: RealNumber,
        kappa: RealNumber,
        w_v_A: RealNumber,
        C: RealNumber,
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

        print(self.datacollector.agent_reporters)

        # Create agents
        IrritabilityAgent.create_agents(
            model=self,
            environment=self.environment,
            S=S,
            V_NR=V_NR,
            V_DE=V_DE,
            V_AB=V_AB,
            V_GR=V_GR,
            M_A=M_A,
            M_S=M_S,
            theta_N_w0=theta_N_w0,
            theta_F_w0=theta_F_w0,
            theta_A_w0=theta_A_w0,
            theta_A_w1=theta_A_w1,
            theta_E_w0=theta_E_w0,
            lambda_A=lambda_A,
            eta=eta,
            gamma=gamma,
            alpha=alpha,
            kappa=kappa,
            w_v_A=w_v_A,
            C=C,
            n=1
        )

    def step(self):
        print(f"model step {self.steps}")

        # Check for terminal state => terminate before action
        states = self.agents.map("get_state")
        if any(s in (IrritabilityAgent.State.GOAL_ABANDONED,
                     IrritabilityAgent.State.GOAL_GRANTED)
               for s in states):
            print("reached a terminal state")
            self.running = False

            # TODO: Set action, reward, etc. to None

            self.datacollector.collect(self)

            # Work-around for the off-by-one bug in batch_run
            self.steps = 3

            return

        print("telling agents to choose_action_and_act()")
        self.agents.do("choose_action_and_act")

        self.datacollector.collect(self)

        print("telling agents to update_emotions_and_learn()")
        self.agents.do("update_emotions_and_learn")

        print("telling agents to transit_state()")
        self.agents.map("transit_state")


if __name__ == "__main__":
    pass
