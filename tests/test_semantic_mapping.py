from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "data"))

from semantic_mapping.common import Lookup, norm_path, norm_scene, norm_taxon
from semantic_mapping.sources import _sun_leaf


def test_lookup_unique_basename():
    lookup = Lookup()
    lookup.add("train/Plantae/42/a.jpg", "Species A", "source-a")
    leaf, method, _ = lookup.resolve("images/a.jpg")
    assert leaf == "Species A"
    assert method == "exact_basename_unique"


def test_lookup_ambiguous_basename_is_not_guessed():
    lookup = Lookup()
    lookup.add("a/class1/0001.jpg", "scene one", "source-1")
    lookup.add("b/class2/0001.jpg", "scene two", "source-2")
    leaf, method, detail = lookup.resolve("images/0001.jpg")
    assert leaf is None
    assert method == "ambiguous_basename"
    assert "scene one" in detail
    assert "scene two" in detail


def test_exact_relative_path_wins():
    lookup = Lookup()
    lookup.add("a/class1/0001.jpg", "scene one", "source-1")
    lookup.add("b/class2/0001.jpg", "scene two", "source-2")
    leaf, method, _ = lookup.resolve("a/class1/0001.jpg")
    assert leaf == "scene one"
    assert method == "exact_relative_path"


def test_normalization_is_conservative():
    assert norm_path("/a/b/c.jpg") == "a/b/c.jpg"
    assert norm_scene("canal (natural)") == "canal natural"
    assert norm_scene("forest_broadleaf") == "forest broadleaf"
    assert norm_taxon("Crocosmia " + chr(215) + " crocosmiiflora") == (
        "crocosmia x crocosmiiflora"
    )


def test_sun_nested_category_path():
    selected = {
        "canal natural": "canal (natural)",
        "badlands": "badlands",
    }
    assert _sun_leaf("/c/canal/natural/sun_abc.jpg", selected) == (
        "canal (natural)"
    )
    assert _sun_leaf("/b/badlands/sun_xyz.jpg", selected) == "badlands"
