
import numpy as np
import pytest
from BITGOT_ETAP1_foundation import MathCore

def test_ichimoku_insufficient_data():
    """
    Test that ichimoku returns arrays of the correct length even when
    the input data is shorter than the window parameters.
    """
    # Create input arrays shorter than the default k=26 or s=52
    length = 10
    high = np.random.rand(length).astype(float)
    low = np.random.rand(length).astype(float)

    # Call ichimoku with default parameters (t=9, k=26, s=52)
    results = MathCore.ichimoku(high, low)

    # Verify all returned arrays have the same length as the input
    for name, arr in results.items():
        assert len(arr) == length, f"Length mismatch for {name}: expected {length}, got {len(arr)}"

def test_ichimoku_exact_k():
    """Test ichimoku when input length is exactly k."""
    k = 26
    length = k
    high = np.random.rand(length).astype(float)
    low = np.random.rand(length).astype(float)

    results = MathCore.ichimoku(high, low, k=k)

    for name, arr in results.items():
        assert len(arr) == length, f"Length mismatch for {name}: expected {length}, got {len(arr)}"
