from kevin import Kevin, Problem
from kevin.human_gate import build_briefing
from kevin.method_library import extract_method


def _problem():
    return Problem(
        statement="how do we make onboarding feel less like paperwork?",
        domain="people-ops",
        constraints=("no extra headcount",),
        known_approaches=("checklists", "a welcome email"),
    )


def test_full_run_produces_a_routed_creative_run():
    run = Kevin().run(_problem())
    assert run.spaces, "should explore spaces"
    assert run.chosen_spaces, "should route to under-explored spaces"
    assert run.variants, "wild brother should produce variants"
    assert run.candidates, "should build candidates"
    assert len(run.evaluations) == len(run.candidates)
    # Only routed spaces are varied.
    varied_spaces = {v.space_id for v in run.variants}
    assert varied_spaces <= set(run.chosen_spaces)


def test_wild_variants_get_method_disciplined():
    run = Kevin().run(_problem(), discipline_threshold=0.7)
    # At least one transfer should have fired for the wild end of the repertoire.
    assert run.transfers, "wild variants above threshold should pull in Layer-9 methods"
    disciplined = [c for c in run.candidates if c.method_disciplined]
    assert disciplined, "expected at least one method-disciplined candidate"


def test_run_is_replay_stable():
    a = Kevin().run(_problem())
    b = Kevin().run(_problem())
    assert a.id == b.id
    assert [e.candidate_id for e in a.evaluations] == [e.candidate_id for e in b.evaluations]
    assert [e.score for e in a.evaluations] == [e.score for e in b.evaluations]


def test_briefing_renders_and_separates_verdicts():
    run = Kevin().run(_problem())
    briefing = build_briefing(run)
    text = briefing.render()
    assert "PROBLEM:" in text
    assert "direction" in text and "taste" in text and "risk" in text and "value" in text
    # promising + tentative listed are subsets of evaluations
    listed = {c.id for _, c in briefing.promising} | {c.id for _, c in briefing.tentative}
    assert listed <= {c.id for c in run.candidates}


def test_loop_closes_extract_method_from_a_real_run():
    kevin = Kevin()
    run = kevin.run(_problem())
    some_candidate = run.candidates[0]
    before = len(kevin.library.all())
    method = extract_method(run, some_candidate.id, name="onboarding_pattern")
    kevin.library.add(method)
    assert len(kevin.library.all()) == before + 1
    assert method.origin == "kevin"
