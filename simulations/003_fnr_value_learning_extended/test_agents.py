import pytest
from unittest.mock import Mock, patch
from agents import IrritabilityAgent


def test_agent_initialization_valid():
    mock_model = Mock()

    agent = IrritabilityAgent(
        model=mock_model,
        V=0,
        M_A=1,
        lambda_A=0.6,
        C=0.8,
        eta=0.1,
        gamma=0.4,
        alpha=0.5,
        kappa=3
    )

    assert agent._variables["V"] == pytest.approx(0)
    assert agent._variables["M_A"] == pytest.approx(1)
    assert agent._variables["lambda_A"] == pytest.approx(0.6)
    assert agent._variables["C"] == pytest.approx(0.8)
    assert agent._variables["eta"] == pytest.approx(0.1)
    assert agent._variables["gamma"] == pytest.approx(0.4)
    assert agent._variables["alpha"] == pytest.approx(0.5)
    assert agent._variables["kappa"] == pytest.approx(3)
    assert agent._variables["r"] is None
    assert agent._variables["rpe"] is None
    assert agent._variables["trial_nr"] == 1
    assert agent._variables["block_nr"] == 1


BASE_KWARGS = dict(
    V=0.0,
    M_A=0.0,
    lambda_A=0.5,
    C=0.5,
    eta=0.5,
    gamma=0.5,
    alpha=0.5,
    kappa=1.0,
)


@pytest.mark.parametrize(
    "override",
    [
        {"C": -0.1},
        {"C": 1.1},
        {"lambda_A": -0.1},
        {"lambda_A": 1.1},
        {"eta": -0.1},
        {"eta": 1.1},
        {"gamma": -0.1},
        {"gamma": 1.1},
        {"alpha": -0.1},
        {"alpha": 1.1},
        {"kappa": 0.0},
        {"kappa": 0.99},
    ],
)
def test_constructor_parameter_constraints(override):
    kwargs = BASE_KWARGS | override

    with pytest.raises(ValueError):
        IrritabilityAgent(
            model=Mock(),
            **kwargs
        )


def test_get_reward_block_1_positive():
    mock_model = Mock()

    agent = IrritabilityAgent(
        model=mock_model,
        V=0,
        M_A=1,
        lambda_A=0.6,
        C=0.8,
        eta=0.1,
        gamma=0.4,
        alpha=0.5,
        kappa=1
    )

    with patch.object(agent.random, "random", return_value=0.97):
        assert agent.get_reward() == 0.5
        assert agent._variables["block_nr"] == 1


def test_get_reward_block_1_negative():
    mock_model = Mock()

    agent = IrritabilityAgent(
        model=mock_model,
        V=0,
        M_A=1,
        lambda_A=0.6,
        C=0.8,
        eta=0.1,
        gamma=0.4,
        alpha=0.5,
        kappa=1
    )

    with patch.object(agent.random, "random", return_value=0.98):
        assert agent.get_reward() == -0.5
        assert agent._variables["block_nr"] == 1


def test_get_reward_block_2_positive():
    mock_model = Mock()

    agent = IrritabilityAgent(
        model=mock_model,
        V=0,
        M_A=1,
        lambda_A=0.6,
        C=0.8,
        eta=0.1,
        gamma=0.4,
        alpha=0.5,
        kappa=1
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
        gamma=0.4,
        alpha=0.5,
        kappa=1
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
        gamma=0.4,
        alpha=0.5,
        kappa=1
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
        gamma=0.4,
        alpha=1.0,
        kappa=1
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
        gamma=0.4,
        alpha=1.0,
        kappa=1
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
        gamma=gamma,
        alpha=1.0,
        kappa=1
    )
    agent._variables["block_nr"] = 1
    agent._variables["trial_nr"] = 1

    with patch.object(agent, "get_reward", return_value=r):
        agent.act()

        assert agent._variables["rpe"] == pytest.approx(expected_rpe)


def make_agent_for_emotions(rpe=0.0, r=0.0, M_A=0.0, lambda_A=0.5, C=1.0,
                            eta=1.0, gamma=1.0, V=0.0, alpha=0.5, kappa=1.0):
    """Helper to create a minimal agent with necessary variables"""
    mock_model = Mock()
    agent = IrritabilityAgent(
        model=mock_model,
        V=V,
        M_A=M_A,
        lambda_A=lambda_A,
        C=C,
        eta=eta,
        gamma=gamma,
        alpha=alpha,
        kappa=kappa)
    agent._variables["r"] = r
    agent._variables["rpe"] = rpe  # not consistent with V but doesn't matter
    return agent


@pytest.mark.parametrize(
    "affective_input, kappa, expected",
    [
        # negative inputs → scaled
        (-1.0, 1.0, -1.0),
        (-1.0, 2.0, -2.0),
        (-0.5, 3.0, -1.5),

        # zero → unchanged
        (0.0, 2.0, 0.0),

        # positive → unchanged
        (0.5, 2.0, 0.5),
        (1.0, 10.0, 1.0),
    ],
)
def test_negativity_bias(affective_input, kappa, expected):
    agent = make_agent_for_emotions(kappa=kappa)

    result = agent.negativity_bias(affective_input)

    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "M_A, r, rpe, alpha, kappa, C, lambda_A, expected",
    [
        # --- No learning (lambda = 1) ---
        (1.0, 1.0, 1.0, 0.5, 2.0, 1.0, 1.0, 1.0),

        # --- No emotional carry-over (lambda = 0) ---
        # affect = 0.5*1 + 0.5*1 = 1 → positive
        (0.0, 1.0, 1.0, 0.5, 2.0, 1.0, 0.0, 1.0),

        # --- Pure RPE (alpha = 1) ---
        (0.0, 0.0, -1.0, 1.0, 2.0, 1.0, 0.0, -2.0),

        # --- Pure reward (alpha = 0) ---
        (0.0, -1.0, 0.0, 0.0, 3.0, 1.0, 0.0, -3.0),

        # --- Mixed input, negative → scaled ---
        # affect = 0.5*(-1) + 0.5*(0) = -0.5 → *2 = -1
        (0.0, 0.0, -1.0, 0.5, 2.0, 1.0, 0.0, -1.0),

        # --- Mixed input, positive → unchanged ---
        # affect = 0.5*(1) + 0.5*(1) = 1
        (0.0, 1.0, 1.0, 0.5, 10.0, 1.0, 0.0, 1.0),

        # --- Scaling by C ---
        # affect = 1 → C=0.5 → 0.5
        (0.0, 1.0, 1.0, 0.5, 1.0, 0.5, 0.0, 0.5),

        # --- Partial update (lambda in (0,1)) ---
        # target = -1 → M_new = 1 + 0.5*(-1 - 1) = 0
        (1.0, 0.0, -1.0, 1.0, 1.0, 1.0, 0.5, 0.0),
    ],
)
def test_update_emotions_parametrized(
    M_A, r, rpe, alpha, kappa, C, lambda_A, expected
):
    agent = make_agent_for_emotions(
        M_A=M_A,
        r=r,
        rpe=rpe,
        alpha=alpha,
        kappa=kappa,
        C=C,
        lambda_A=lambda_A,
    )

    agent.update_emotions()

    assert agent._variables["M_A"] == pytest.approx(expected)
