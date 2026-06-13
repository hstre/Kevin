import pytest

from kevin import Problem, desi_link
from kevin.space_predictor import SpacePredictor

if not desi_link.available():
    pytest.skip("DESi not available (set DESI_ROOT or install desi-governance)",
                allow_module_level=True)


def test_disabled_without_flag(monkeypatch):
    monkeypatch.delenv("KEVIN_USE_DESI", raising=False)
    assert desi_link.enabled() is False
    assert desi_link.coverage({0, 1}, {0, 1, 2, 3}) is None
    assert desi_link.engine() == "kevin-builtin"


def test_enabled_uses_real_desi_coverage(monkeypatch):
    monkeypatch.setenv("KEVIN_USE_DESI", "1")
    assert desi_link.enabled() is True
    report = desi_link.coverage({0, 1, 2}, set(range(8)))
    assert report is not None
    assert report["engine"] == "DESi"
    assert report["blindspot_count"] == 5            # 8 - 3 worked
    assert report["new_region_fraction"] == 0.625    # 5/8 unexplored


def test_predictor_uses_desi_when_enabled(monkeypatch):
    monkeypatch.setenv("KEVIN_USE_DESI", "1")
    pred = SpacePredictor().predict(Problem("reduce friction in code review"))
    assert pred.engine == "DESi"
    assert pred.blindspots
