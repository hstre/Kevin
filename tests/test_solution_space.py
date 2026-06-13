from kevin.llm_client import MockLLM
from kevin.models import Problem
from kevin.solution_space import SolutionSpaceExplorer


def _problem(**kw):
    return Problem(statement="how do we make onboarding feel less like paperwork?", **kw)


def test_explore_returns_scored_spaces_sorted_by_opportunity():
    exp = SolutionSpaceExplorer(MockLLM())
    spaces = exp.explore(_problem())
    assert spaces, "explorer should propose spaces"
    opportunities = [s.opportunity for s in spaces]
    assert opportunities == sorted(opportunities, reverse=True)
    for s in spaces:
        assert 0.0 <= s.plausibility <= 1.0
        assert 0.0 <= s.exploration <= 1.0
        assert s.opportunity == round(s.plausibility * (1 - s.exploration), 4)


def test_known_approaches_raise_exploration_and_lower_opportunity():
    exp = SolutionSpaceExplorer(MockLLM())
    # A known approach that overlaps the "actor / interests" region vocabulary.
    naive = exp.explore(_problem())
    informed = exp.explore(_problem(known_approaches=("whose interests shape the evidence",)))
    naive_by_axis = {s.axis: s for s in naive}
    informed_by_axis = {s.axis: s for s in informed}
    # The 'actor' region should look more explored once a matching approach is known.
    assert informed_by_axis["actor"].exploration >= naive_by_axis["actor"].exploration


def test_routing_picks_top_underexplored():
    exp = SolutionSpaceExplorer(MockLLM())
    picked = exp.select_underexplored(_problem(), top_k=2)
    assert 1 <= len(picked) <= 2
    ranked = exp.explore(_problem())
    assert picked[0].id == ranked[0].id


def test_routing_never_empty_when_spaces_exist():
    exp = SolutionSpaceExplorer(MockLLM())
    picked = exp.select_underexplored(_problem(), top_k=2, min_opportunity=2.0)
    assert len(picked) == 1  # falls back to best even if nothing clears the bar


def test_replay_stable_ids():
    exp = SolutionSpaceExplorer(MockLLM())
    a = exp.explore(_problem())
    b = exp.explore(_problem())
    assert [s.id for s in a] == [s.id for s in b]
