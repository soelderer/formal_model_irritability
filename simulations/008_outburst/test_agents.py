import pytest
from unittest.mock import Mock, patch
from agents import IrritabilityAgent


def test_agent_initialization_valid():
    mock_model = Mock()

    agent = IrritabilityAgent(
        model=mock_model,
        V=0,
        M_A=1,
        M_S=0.8,
        lambda_A=0.6,
        eta=0.1,
        gamma=0.4,
        alpha=0.5,
        kappa=3,
        lambda_C=0.5,
        midpoint=150
    )

    assert agent._variables["V"] == pytest.approx(0)
    assert agent._variables["M_A"] == pytest.approx(1)
    assert agent._variables["M_S"] == pytest.approx(0.8)
    assert agent._variables["lambda_A"] == pytest.approx(0.6)
    assert agent._variables["eta"] == pytest.approx(0.1)
    assert agent._variables["gamma"] == pytest.approx(0.4)
    assert agent._variables["alpha"] == pytest.approx(0.5)
    assert agent._variables["kappa"] == pytest.approx(3)
    assert agent._variables["r"] is None
    assert agent._variables["rpe"] is None
    assert agent._variables["C"] is None
    assert agent._variables["lambda_C"] == pytest.approx(0.5)
    assert agent._variables["midpoint"] == 150
    assert agent._variables["trial_nr"] == 1
    assert agent._variables["block_nr"] == 1


BASE_KWARGS = dict(
    V=0.0,
    M_A=0.0,
    M_S=0.0,
    lambda_A=0.5,
    eta=0.5,
    gamma=0.5,
    alpha=0.5,
    kappa=1.0,
    lambda_C=0.5,
    midpoint=150
)


@pytest.mark.parametrize(
    "override",
    [
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
        {"lambda_C": -0.01},
        {"lambda_C": 2},
        {"midpoint": 99},
        {"midpoint": 201},
    ],
)
def test_constructor_parameter_constraints(override):
    kwargs = BASE_KWARGS | override

    with pytest.raises(ValueError):
        IrritabilityAgent(
            model=Mock(),
            **kwargs
        )


BASE_KWARGS = dict(
    V=0,
    M_A=1,
    M_S=1,
    lambda_A=0.6,
    eta=0.1,
    gamma=0.4,
    alpha=0.5,
    kappa=1,
    lambda_C=0.5,
    midpoint=150,
)


@pytest.mark.parametrize(
    "block_nr, rand, expected",
    [
        # block 1
        (1, 0.97, 0.5),
        (1, 0.98, -0.5),

        # block 2
        (2, 0.39, 0.5),
        (2, 0.40, -0.5),
    ],
)
def test_get_reward(block_nr, rand, expected):
    agent = IrritabilityAgent(model=Mock(), **BASE_KWARGS)
    agent._variables["block_nr"] = block_nr

    with patch.object(agent.random, "random", return_value=rand):
        assert agent.get_reward() == expected


def test_get_reward_invalid_block_raises():
    agent = IrritabilityAgent(model=Mock(), **BASE_KWARGS)
    agent._variables["block_nr"] = 3

    with pytest.raises(ValueError):
        agent.get_reward()


@pytest.mark.parametrize(
    "trial_nr, expected_block, expected_trial",
    [
        (1, 1, 2),      # normal progression
        (99, 2, 100),   # block transition
    ],
)
def test_act_block_1(trial_nr, expected_block, expected_trial):
    agent = IrritabilityAgent(
        model=Mock(),
        V=1,
        M_A=1,
        M_S=1,
        lambda_A=0.6,
        eta=0.1,
        gamma=0.4,
        alpha=1.0,
        kappa=1,
        lambda_C=0.5,
        midpoint=150,
    )

    agent._variables["block_nr"] = 1
    agent._variables["trial_nr"] = trial_nr

    with patch.object(agent, "get_reward", return_value=0.5):
        agent.act()

    assert agent._variables["r"] == pytest.approx(0.5)
    assert agent._variables["rpe"] == pytest.approx(-0.1)
    assert agent._variables["block_nr"] == expected_block
    assert agent._variables["trial_nr"] == expected_trial


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
        M_S=0.0,
        lambda_A=0.5,
        eta=0.1,
        gamma=gamma,
        alpha=1.0,
        kappa=1,
        lambda_C=0.5,
        midpoint=150
    )
    agent._variables["block_nr"] = 1
    agent._variables["trial_nr"] = 1

    with patch.object(agent, "get_reward", return_value=r):
        agent.act()

        assert agent._variables["rpe"] == pytest.approx(expected_rpe)


@pytest.mark.parametrize(
    "trial_nr, expected_range",
    [
        (0, (0.993, 1.0)),       # early trial
        (50, (0.993, 0.995)),    # before midpoint
        (150, (0.5, 0.51)),      # around midpoint
        (300, (0.0002, 0.01)),   # very late
    ],
)
def test_controllability_block_2_logistic_decay(trial_nr, expected_range):
    agent = IrritabilityAgent(
        model=Mock(),
        V=0,
        M_A=0,
        M_S=0,
        lambda_A=0.5,
        eta=0.1,
        gamma=0.9,
        alpha=0.5,
        kappa=1.0,
        lambda_C=0.05,
        midpoint=150,
    )

    agent._variables["block_nr"] = 2
    agent._variables["trial_nr"] = trial_nr

    C = agent.get_controllability()
    low, high = expected_range

    assert low <= C <= high


def make_agent_for_emotions(rpe=0.0, r=0.0, M_A=0.0, M_S=0.0, lambda_A=0.5,
                            eta=1.0, gamma=1.0, V=0.0, alpha=0.5, kappa=1.0,
                            lambda_C=0.5, midpoint=150, C=0.5):
    """Helper to create a minimal agent with necessary variables"""
    mock_model = Mock()
    agent = IrritabilityAgent(
        model=mock_model,
        V=V,
        M_A=M_A,
        M_S=M_S,
        lambda_A=lambda_A,
        eta=eta,
        gamma=gamma,
        alpha=alpha,
        kappa=kappa,
        lambda_C=lambda_C,
        midpoint=midpoint)
    agent._variables["r"] = r
    agent._variables["rpe"] = rpe  # not consistent with V but doesn't matter
    agent._variables["C"] = C  # not consistent with logistic decay
    return agent


@pytest.mark.parametrize(
    "C, M_A_init, M_S_init, lambda_A, alpha, kappa, r, rpe, expected_MA,"
    "expected_MS",
    [
        # Anger full, sadness zero
        (1.0, 0.0, 0.0, 0.5, 0.5, 1.0, 0.5, 0.0, 0.125, 0.0),
        # Sadness full, anger zero
        (0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 0.5, 0.0, 0.0, 0.125),
        # Half-half weighting
        (0.5, 0.1, 0.2, 0.5, 0.5, 1.0, 0.4, 0.0, 0.1, 0.15),
        # Negative affective input triggers kappa
        (0.5, 0.0, 0.0, 0.5, 1.0, 2.0, -0.4, -0.1, -0.05, -0.05),
        # Edge: initial emotions already high
        (0.5, 1.0, 0.5, 0.5, 0.5, 1.0, 0.2, 0.0, 0.525, 0.275),
    ],
)
def test_emotion_updates_r_rpe(C, M_A_init, M_S_init, lambda_A, alpha, kappa,
                               r, rpe, expected_MA, expected_MS):

    agent = make_agent_for_emotions(
        r=r,
        rpe=rpe,
        M_A=M_A_init,
        lambda_A=lambda_A,
        C=C,
        alpha=alpha,
        kappa=kappa
    )

    affective = alpha * \
        agent._variables["rpe"] + (1-alpha) * agent._variables["r"]
    if affective < 0:
        affective *= kappa

    agent._variables["M_A"] = M_A_init + \
        (1-lambda_A) * (C * affective - M_A_init)
    agent._variables["M_S"] = M_S_init + \
        (1-lambda_A) * ((1-C) * affective - M_S_init)

    assert agent._variables["M_A"] == pytest.approx(expected_MA, rel=1e-3)
    assert agent._variables["M_S"] == pytest.approx(expected_MS, rel=1e-3)


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
