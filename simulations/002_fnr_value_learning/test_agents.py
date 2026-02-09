import pytest
from unittest.mock import Mock, patch
from agents import IrritabilityAgent
import mesa
import numpy as np

#     def __init__(
#         self,
#         model: mesa.Model,
#         V: RealNumber,
#         M_A: RealNumber,
#         lambda_A: RealNumber,
#         C: RealNumber,
#         eta: RealNumber,
#         gamma: RealNumber

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
