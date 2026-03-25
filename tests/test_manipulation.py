import pytest
import sys
import unittest.mock as mock
import math

# Mocking numpy so np.clip works locally without requiring numpy
class NumpyMock:
    @staticmethod
    def clip(a, a_min, a_max):
        return max(a_min, min(a, a_max))

sys.modules['ccxt'] = mock.MagicMock()
sys.modules['ccxt.async_support'] = mock.MagicMock()
np_mock = mock.MagicMock()
np_mock.clip = NumpyMock.clip
sys.modules['numpy'] = np_mock
sys.modules['pandas'] = mock.MagicMock()
sys.modules['torch'] = mock.MagicMock()
sys.modules['torch.nn'] = mock.MagicMock()
sys.modules['torch.optim'] = mock.MagicMock()

from BITGOT_ETAP1_foundation import detect_manipulation, StateVector, _safe_check, _MANIPULATION_CHECKS

def test_detect_manipulation_empty():
    sv = StateVector()
    risk, detected = detect_manipulation(sv)
    assert detected == []
    assert risk == 0.0

def test_detect_manipulation_pump_dump():
    sv = StateVector()
    # "pump_dump",     lambda sv: sv.pc_1m > 0.04 and sv.taker_ratio > 0.80 and sv.funding < 0
    sv.pc_1m = 0.05
    sv.taker_ratio = 0.85
    sv.funding = -0.01

    risk, detected = detect_manipulation(sv)
    assert "pump_dump" in detected
    assert risk > 0.0
    # Floating point math can be imprecise, so test using math.isclose
    expected_risk = min(1.0, len(detected) * 0.18 + sv.manipulation_risk() * 0.5)
    assert math.isclose(risk, expected_risk)

def test_detect_manipulation_bear_trap():
    sv = StateVector()
    # "bear_trap",     lambda sv: sv.ob_imb > 0.5 and sv.funding < -0.002 and sv.rsi_14 > 0.75
    sv.ob_imb = 0.6
    sv.funding = -0.003
    sv.rsi_14 = 0.8

    risk, detected = detect_manipulation(sv)
    assert "bear_trap" in detected
    assert risk > 0.0
    expected_risk = min(1.0, len(detected) * 0.18 + sv.manipulation_risk() * 0.5)
    assert math.isclose(risk, expected_risk)

def test_detect_manipulation_multiple():
    sv = StateVector()
    # Trigger both pump_dump and bear_trap
    sv.pc_1m = 0.05
    sv.taker_ratio = 0.85
    sv.funding = -0.01  # < -0.002
    sv.ob_imb = 0.6
    sv.rsi_14 = 0.8

    risk, detected = detect_manipulation(sv)
    assert "pump_dump" in detected
    assert "bear_trap" in detected
    assert len(detected) >= 2
    expected_risk = min(1.0, len(detected) * 0.18 + sv.manipulation_risk() * 0.5)
    assert math.isclose(risk, expected_risk)

def test_detect_manipulation_safe_check():
    class BadStateVector:
        def __getattr__(self, name):
            raise ValueError("Intentional crash")
        def manipulation_risk(self):
            return 0.0

    bad_sv = BadStateVector()
    risk, detected = detect_manipulation(bad_sv)
    assert detected == []
    assert risk == 0.0

def test_detect_manipulation_max_risk():
    sv = StateVector()
    # Max manipulation risk from sv:
    # toxic_flow > 0.6: +0.3
    # vpin > 0.7: +0.2
    # abs(funding) > 0.003: +0.2
    # liq_cascade > 0.6: +0.2
    # ob_entropy < 0.3: +0.1
    # total = 1.0 (manipulation_risk caps at 1.0)
    sv.toxic_flow = 0.7
    sv.vpin = 0.8
    sv.funding = 0.004
    sv.liq_cascade = 0.7
    sv.ob_entropy = 0.2

    # Verify manipulation_risk is roughly 1.0
    assert math.isclose(sv.manipulation_risk(), 1.0)

    # Trigger patterns to push total risk over 1.0
    # 1. Pump Dump
    sv.pc_1m = 0.05
    sv.taker_ratio = 0.85
    sv.funding = -0.01

    # 2. Bear Trap
    sv.ob_imb = 0.6
    sv.rsi_14 = 0.8

    # 3. Flash Event
    sv.pc_1m = 0.06

    risk, detected = detect_manipulation(sv)
    assert len(detected) >= 3
    assert risk == 1.0
