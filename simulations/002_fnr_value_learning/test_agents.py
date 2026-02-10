import pytest
from unittest.mock import Mock, patch
from agents import IrritabilityAgent
import mesa
import numpy as np

def test_agent_initialization_valid():
    mock_model = Mock()

    agent = IrritabilityAgent(
        model=mock_model,
        V=0,
        M_A=1,
        lambda_A=0.6,
        C=0.8,
        eta=0.1,
        gamma=0.4
    )

    assert agent._variables["V"] == pytest.approx(0)
    assert agent._variables["M_A"] == pytest.approx(1)
    assert agent._variables["lambda_A"] == pytest.approx(0.6)
    assert agent._variables["C"] == pytest.approx(0.8)
    assert agent._variables["eta"] == pytest.approx(0.1)
    assert agent._variables["gamma"] == pytest.approx(0.4)
    assert agent._variables["r"] is None
    assert agent._variables["rpe"] is None
    assert agent._variables["trial_nr"] is 1
    assert agent._variables["block_nr"] is 1

def test_get_reward_block_1_positive():
    mock_model = Mock()

    agent = IrritabilityAgent(
        model=mock_model,
        V=0,
        M_A=1,
        lambda_A=0.6,
        C=0.8,
        eta=0.1,
        gamma=0.4
    )

    with patch.object(agent.random, "random", return_value=0.97):
        assert agent.get_reward() == 0.5
        assert agent._variables["block_nr"] is 1

def test_get_reward_block_1_negative():
    mock_model = Mock()

    agent = IrritabilityAgent(
        model=mock_model,
        V=0,
        M_A=1,
        lambda_A=0.6,
        C=0.8,
        eta=0.1,
        gamma=0.4
    )

    with patch.object(agent.random, "random", return_value=0.98):
        assert agent.get_reward() == -0.5
        assert agent._variables["block_nr"] is 1

def test_get_reward_block_2_positive():
    mock_model = Mock()

    agent = IrritabilityAgent(
        model=mock_model,
        V=0,
        M_A=1,
        lambda_A=0.6,
        C=0.8,
        eta=0.1,
        gamma=0.4
    )

    agent._variables["block_nr"] = 2

    with patch.object(agent.random, "random", return_value=0.39):
        assert agent.get_reward() == 0.5

def test_get_reward_block_2_negative():
    mock_model = Mock()

    agent = IrritabilityAgent(
        model=mock_model,
        V=0,
        M_A=1,
        lambda_A=0.6,
        C=0.8,
        eta=0.1,
        gamma=0.4
    )
    agent._variables["block_nr"] = 2

    with patch.object(agent.random, "random", return_value=0.40):
        assert agent.get_reward() == -0.5

def test_get_reward_block_3_raises():
    mock_model = Mock()

    agent = IrritabilityAgent(
        model=mock_model,
        V=0,
        M_A=1,
        lambda_A=0.6,
        C=0.8,
        eta=0.1,
        gamma=0.4
    )
    agent._variables["block_nr"] = 3

    with pytest.raises(ValueError):
        agent.get_reward()

def test_act_block_1():
    mock_model = Mock()

    agent = IrritabilityAgent(
        model=mock_model,
        V=1,
        M_A=1,
        lambda_A=0.6,
        C=0.8,
        eta=0.1,
        gamma=0.4
    )
    agent._variables["block_nr"] = 1
    agent._variables["trial_nr"] = 1

    with patch.object(agent, "get_reward", return_value=0.5):
        agent.act()

        # rpe = r + gamma * V - V = 0.5 + 0.4 * 1 - 1
        assert agent._variables["r"] == pytest.approx(0.5)
        assert agent._variables["rpe"] == pytest.approx(-0.1)

        assert agent._variables["block_nr"] == 1
        assert agent._variables["trial_nr"] == 2

def test_act_block_1_trial_99():
    mock_model = Mock()

    agent = IrritabilityAgent(
        model=mock_model,
        V=1,
        M_A=1,
        lambda_A=0.6,
        C=0.8,
        eta=0.1,
        gamma=0.4
    )
    agent._variables["block_nr"] = 1
    agent._variables["trial_nr"] = 99

    with patch.object(agent, "get_reward", return_value=0.5):
        agent.act()

        # rpe = r + gamma * V - V = 0.5 + 0.4 * 1 - 1
        assert agent._variables["r"] == pytest.approx(0.5)
        assert agent._variables["rpe"] == pytest.approx(-0.1)

        assert agent._variables["block_nr"] == 2
        assert agent._variables["trial_nr"] == 100

@pytest.mark.parametrize(
    "V, r, gamma, expected_rpe",
    [
        (1.0, 0.5, 0.4, -0.1),
        (1.0, 0.5, 0.0, -0.5),
        (1.0, 0.5, 1.0, 0.5),
        (0.0, 0.5, 0.4, 0.5),
        (1.0, -0.5, 0.4, -1.1),
    ],
)
def test_act_rpe_parametrized(V, r, gamma, expected_rpe):
    agent = IrritabilityAgent(
        model=Mock(),
        V=V,
        M_A=0.0,
        lambda_A=0.5,
        C=1.0,
        eta=0.1,
        gamma=gamma
    )
    agent._variables["block_nr"] = 1
    agent._variables["trial_nr"] = 1

    with patch.object(agent, "get_reward", return_value=r):
        agent.act()

        assert agent._variables["rpe"] == pytest.approx(expected_rpe)

def make_agent_for_emotions(rpe=0.0, M_A=0.0, lambda_A=0.5, C=1.0,
                            eta=1.0, gamma=1.0, V=0.0):
    """Helper to create a minimal agent with necessary variables"""
    mock_model = Mock()
    agent = IrritabilityAgent(
        model=mock_model,
        V=V,
        M_A=M_A,
        lambda_A=lambda_A,
        C=C,
        eta=eta,
        gamma=gamma
    )
    agent._variables["rpe"] = rpe
    return agent

def test_update_emotions_basic():
    agent = make_agent_for_emotions(rpe=2.0, M_A=1.0, lambda_A=0.5, C=1.0)
    agent.update_emotions()

    # M_A_new = 1 + (1-0.5)*(1*2 - 1) = 1 + 0.5*(1) = 1.5
    assert agent._variables["M_A"] == pytest.approx(1.5)

def test_update_emotions_lambda_zero():
    agent = make_agent_for_emotions(rpe=2.0, M_A=1.0, lambda_A=0.0, C=1.0)
    agent.update_emotions()

    # M_A_new = 1 + (1-0)*(1*2 -1) = 1 + 1*(1) = 2
    assert agent._variables["M_A"] == pytest.approx(2.0)

def test_update_emotions_lambda_one():
    agent = make_agent_for_emotions(rpe=2.0, M_A=1.0, lambda_A=1.0, C=1.0)
    agent.update_emotions()

    # M_A_new = 1 + (1-1)*(1*2-1) = 1 + 0*(1) = 1
    assert agent._variables["M_A"] == pytest.approx(1.0)

def test_update_emotions_zero_rpe():
    agent = make_agent_for_emotions(rpe=0.0, M_A=1.0, lambda_A=0.5, C=1.0)
    agent.update_emotions()

    # M_A_new = 1 + 0.5*(1*0 -1) = 1 + 0.5*(-1) = 0.5
    assert agent._variables["M_A"] == pytest.approx(0.5)

def test_update_emotions_zero_C():
    agent = make_agent_for_emotions(rpe=2.0, M_A=1.0, lambda_A=0.5, C=0.0)
    agent.update_emotions()

    # M_A_new = 1 + 0.5*(0*2 -1) = 1 + 0.5*(-1) = 0.5
    assert agent._variables["M_A"] == pytest.approx(0.5)

@pytest.mark.parametrize(
    "V, rpe, eta, expected_V",
    [
        # No learning signal
        (1.0, 0.0, 0.1, 1.0),

        # No learning rate
        (1.0, 2.0, 0.0, 1.0),

        # Standard positive update
        (1.0, 2.0, 0.1, 1.2),

        # Negative prediction error
        (1.0, -2.0, 0.1, 0.8),

        # Negative value estimate
        (-1.0, 2.0, 0.1, -0.8),

        # Large learning rate (full correction)
        (1.0, 2.0, 1.0, 3.0),

        # Small learning rate
        (1.0, 2.0, 1e-6, 1.000002),
    ],
)
def test_learn_state_value_parametrized(V, rpe, eta, expected_V):
    agent = make_agent_for_emotions(rpe=rpe, V=V, eta=eta)
    agent.learn_state_value()
    assert agent._variables["V"] == pytest.approx(expected_V)
