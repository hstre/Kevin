from kevin import Problem
from kevin.space_predictor import STRUCTURAL_AXES, SpacePredictor


def _problem(**kw):
    return Problem("how do we make onboarding feel less like paperwork?", **kw)


def test_predicts_blind_spots_over_the_full_universe():
    pred = SpacePredictor().predict(_problem())
    assert pred.universe_size == len(STRUCTURAL_AXES)
    assert pred.blindspots                      # some axes are open
    assert 0.0 <= pred.new_region_fraction <= 1.0
    # blind spots + covered partition the universe
    assert len(pred.blindspots) + len(pred.covered) == pred.universe_size
    # a seed space is produced for every predicted-open axis
    assert len(pred.seed_spaces) == len(pred.blindspots)


def test_known_approaches_cover_axes_and_shrink_open_space():
    naive = SpacePredictor().predict(_problem())
    informed = SpacePredictor().predict(
        _problem(known_approaches=("a welcome email", "checklists and forms"))
    )
    # Declaring worked approaches should cover >= as many axes (less open space).
    assert len(informed.covered) >= len(naive.covered)
    assert informed.new_region_fraction <= naive.new_region_fraction


def test_seed_spaces_are_under_explored_and_plausible():
    pred = SpacePredictor().predict(_problem())
    for space in pred.seed_spaces:
        assert space.exploration <= 0.3          # predicted-open -> low exploration
        assert space.opportunity > 0.3           # therefore high opportunity
        assert space.affinities                  # carries affinities for method matching


def test_builtin_engine_by_default_and_deterministic():
    a = SpacePredictor().predict(_problem(known_approaches=("checklists",)))
    b = SpacePredictor().predict(_problem(known_approaches=("checklists",)))
    assert a.engine == "kevin-builtin"           # KEVIN_USE_DESI not set
    assert a.to_dict() == b.to_dict()            # replay-stable
