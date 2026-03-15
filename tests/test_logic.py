import pandas as pd
import pytest
from scripts.labels import recalculate_4class_label, EPSILON

def test_epsilon_winner_logic():
    # Scenario 1: Exact Tie in Base/Alt Winners (0 base, 0 alt), but some 'both'
    # alt=0, base=0, both=3, none=2
    row = pd.Series({
        "attr_name_winner": "both",
        "attr_phone_winner": "both",
        "attr_web_winner": "both",
        "attr_address_winner": "none",
        "attr_category_winner": "none"
    })
    assert recalculate_4class_label(row) == "both"

    # Scenario 2: Majority None (>= 3)
    row = pd.Series({
        "attr_name_winner": "alt",
        "attr_phone_winner": "base",
        "attr_web_winner": "none",
        "attr_address_winner": "none",
        "attr_category_winner": "none"
    })
    assert recalculate_4class_label(row) == "none"

    # Scenario 3: Winner wins when n_both is low
    # alt=2, base=1, both=1, none=1. delta=1. n_both=1. 1 is not > 1. 
    # So returns alt.
    row = pd.Series({
        "attr_name_winner": "alt",
        "attr_phone_winner": "alt",
        "attr_web_winner": "base",
        "attr_address_winner": "both",
        "attr_category_winner": "none"
    })
    assert recalculate_4class_label(row) == "alt"

    # Scenario 4: Both wins when n_both > delta
    # alt=2, base=1, both=2, none=0. delta=1. n_both=2. 2 > 1.
    # Returns both.
    row = pd.Series({
        "attr_name_winner": "alt",
        "attr_phone_winner": "alt",
        "attr_web_winner": "base",
        "attr_address_winner": "both",
        "attr_category_winner": "both"
    })
    assert recalculate_4class_label(row) == "both"

if __name__ == "__main__":
    pytest.main([__file__])
