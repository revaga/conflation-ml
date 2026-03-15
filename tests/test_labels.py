import pytest
import pandas as pd
from scripts.labels import recalculate_4class_label, fourclass_to_threeclass, FOUR_TO_THREE, ATTR_ATTRS
from scripts.xgboost_multiclass import MAP_TO_IDX, IDX_TO_MAP

def test_labels_vocabulary():
    """Verify that the vocabulary matches between the base labels and the 4-class multiclass model."""
    assert set(MAP_TO_IDX.keys()) == set(FOUR_TO_THREE.keys())
    assert set(IDX_TO_MAP.values()) == set(FOUR_TO_THREE.keys())

def test_recalculate_logic():
    # Test 'none' logic -> >= 3 none
    row = pd.Series({f"attr_{a}_winner": "none" for a in ATTR_ATTRS})
    assert recalculate_4class_label(row) == "none"

    # Test exact tie in winners -> both
    row = pd.Series({
        "attr_name_winner": "alt", 
        "attr_phone_winner": "base", 
        "attr_web_winner": "both", 
        "attr_address_winner": "none", 
        "attr_category_winner": "none"
    })
    assert recalculate_4class_label(row) == "both"
    
def test_4to3_mapping():
    assert fourclass_to_threeclass("alt") == "alt"
    assert fourclass_to_threeclass("base") == "base"
    assert fourclass_to_threeclass("both") == "both"
    assert fourclass_to_threeclass("none") == "base"
