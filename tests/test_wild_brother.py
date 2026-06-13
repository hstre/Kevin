from kevin.llm_client import MockLLM
from kevin.models import Affinity, Problem, SolutionSpace, WildMove
from kevin.wild_brother import WILDNESS, WildBrother


def _space():
    return SolutionSpace(
        label="Analogy space",
        description="explore via a distant field with the same shape",
        axis="analogy",
        affinities=(Affinity.ANALOGY,),
    )


def test_vary_produces_one_variant_per_move():
    wb = WildBrother(MockLLM())
    moves = (WildMove.ANALOGY, WildMove.RISKY_HYPOTHESIS)
    variants = wb.vary(Problem("p statement here"), _space(), moves=moves)
    assert [v.move for v in variants] == list(moves)
    for v in variants:
        assert v.content
        assert v.wildness == WILDNESS[v.move]


def test_wildness_ordering_is_sane():
    assert WILDNESS[WildMove.ANALOGY] < WILDNESS[WildMove.RISKY_HYPOTHESIS]
    assert WILDNESS[WildMove.ABSURD_COMBINATION] >= WILDNESS[WildMove.WHAT_IF]


def test_storm_covers_all_spaces():
    wb = WildBrother(MockLLM())
    spaces = [_space(), SolutionSpace(label="Boundary", description="edges", axis="boundary")]
    variants = wb.storm(Problem("p statement here"), spaces)
    seen_spaces = {v.space_id for v in variants}
    assert seen_spaces == {s.id for s in spaces}


def test_variants_are_replay_stable():
    wb = WildBrother(MockLLM())
    a = wb.vary(Problem("stable problem"), _space())
    b = wb.vary(Problem("stable problem"), _space())
    assert [v.id for v in a] == [v.id for v in b]
    assert [v.content for v in a] == [v.content for v in b]
