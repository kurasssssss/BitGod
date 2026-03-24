import pytest
from BITGOT_ETAP1_foundation import BotTier

def test_bot_tier_n_engines_all_cases():
    """Test that n_engines returns the correct engine count for all BotTier cases."""
    assert BotTier.APEX.n_engines() == 25
    assert BotTier.ELITE.n_engines() == 15
    assert BotTier.STANDARD.n_engines() == 8
    assert BotTier.SCOUT.n_engines() == 3

    # Ensure all enum members are tested by iterating over them
    for tier in BotTier:
        assert isinstance(tier.n_engines(), int)
        assert tier.n_engines() > 0
