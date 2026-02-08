import pytest
from unittest.mock import Mock
from agents import IrritabilityAgent
import mesa

def test_agent_initialization_no_anger_valid():
    mock_model = Mock()

    agent = IrritabilityAgent(
        model=mock_model,
        V=0.5,
        M_A=0,
        theta_N_w0=3,
        theta_N=None,
        theta_F_w0=-10,
        theta_F=None,
        theta_A_w0=9,
        theta_A_w1=10,
        theta_A=None,
        p_N=None,
        p_F=None,
        p_A=None,
        a=None,
        r=None,
        rpe=None,
        lambda_A=0.5,
        C=1.0
    )

    # Test initialization of numeric variables
    assert agent._variables["V"] == pytest.approx(0.5)
    assert agent._variables["M_A"] == pytest.approx(0.0)
    assert agent._variables["theta_N_w0"] == pytest.approx(3.0)
    assert agent._variables["theta_F_w0"] == pytest.approx(-10.0)
    assert agent._variables["theta_A_w0"] == pytest.approx(9.0)
    assert agent._variables["theta_A_w1"] == pytest.approx(10.0)

    # Quick check on model-determined variables
    assert "theta_A" in agent._variables
    assert "theta_F" in agent._variables
    assert "theta_N" in agent._variables
    assert "p_A" in agent._variables
    assert "p_F" in agent._variables
    assert "p_N" in agent._variables

    # Test other optional variables
    assert agent._variables["r"] is None
    assert agent._variables["rpe"] is None

    # Test constrained parameters
    assert agent._variables["lambda_A"] == pytest.approx(0.5)
    assert agent._variables["C"] == pytest.approx(1.0)

@pytest.mark.parametrize(
    "kwargs",
    [
        {"C": -0.1},
        {"C": 1.1},
        {"lambda_A": -0.5},
        {"lambda_A": 2.0},
    ],
)
def test_probability_like_parameters_out_of_bounds(kwargs):
    with pytest.raises(ValueError):
        IrritabilityAgent(model=Mock(), **kwargs)

def test_calculate_action_tendencies_no_anger():
    mock_model = Mock()

    agent = IrritabilityAgent(
        model=mock_model,
        M_A=0,
        theta_N_w0=3,
        theta_F_w0=-10,
        theta_A_w0=9,
        theta_A_w1=10,  # is irrelevant here
    )

    # Test softmax probabilities with appropriate tolerance
    assert agent._variables["theta_N"] == pytest.approx(3.0)
    assert agent._variables["theta_F"] == pytest.approx(-10.0)
    assert agent._variables["theta_A"] == pytest.approx(9.0)
    assert agent._variables["p_N"] == pytest.approx(0.0024726, rel=1e-3)
    assert agent._variables["p_F"] == pytest.approx(0.0, abs=1e-8)
    assert agent._variables["p_A"] == pytest.approx(0.99753, rel=1e-3)

def test_calculate_action_tendencies_anger():
    mock_model = Mock()

    agent = IrritabilityAgent(
        model=mock_model,
        M_A=-1,
        theta_N_w0=3,
        theta_F_w0=-2,
        theta_A_w0=2,
        theta_A_w1=-2,
    )

    # Test softmax probabilities with appropriate tolerance
    assert agent._variables["theta_N"] == pytest.approx(3.0)
    assert agent._variables["theta_F"] == pytest.approx(-2.0)
    assert agent._variables["theta_A"] == pytest.approx(4)

    # Softmax probabilities
    assert agent._variables["p_N"] == pytest.approx(0.26845495, abs=1e-8)
    assert agent._variables["p_F"] == pytest.approx(0.00180884, abs=1e-8)
    assert agent._variables["p_A"] == pytest.approx(0.72973621, rel=1e-6)
