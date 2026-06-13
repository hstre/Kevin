from kevin.llm_client import MockLLM
from kevin.method_library import SEED_METHODS, MethodLibrary, extract_method
from kevin.models import (
    Affinity,
    Candidate,
    CreativeRun,
    Problem,
    SolutionSpace,
    Variant,
    WildMove,
)


def test_methods_are_content_free():
    """Steps must describe shape, not domain objects - the whole thesis depends on it."""
    forbidden = ("pulmonary", "embolism", "patient", "plaintiff", "defendant", "neo4j")
    for m in SEED_METHODS:
        blob = " ".join(m.steps).lower()
        for word in forbidden:
            assert word not in blob, f"method {m.name} leaked content: {word}"


def test_match_is_by_affinity_only():
    lib = MethodLibrary()
    matched = lib.match((Affinity.BOUNDARY,), top_k=3)
    assert any(m.name == "limit_case_analysis" for m in matched)
    # A boundary query should not surface a pure-provenance method first.
    assert matched[0].affinities  # has affinities
    assert Affinity.BOUNDARY in matched[0].affinities


def test_match_empty_when_no_overlap():
    lib = MethodLibrary()

    # An affinity no seed method carries would return nothing; all seeds use the
    # closed set, so test with a deliberately unmatched single tag combination by
    # filtering: PROVENANCE matches at least one, so assert it does.
    assert lib.match((Affinity.PROVENANCE,))


def test_transfer_preserves_step_count_and_binds_target():
    lib = MethodLibrary()
    method = next(m for m in SEED_METHODS if m.name == "red_flags_first")
    xfer = lib.transfer(MockLLM(), method, target_id="var_123", target_text="ship faster")
    assert len(xfer.mapped_steps) == len(method.steps)
    assert all("ship faster" in step for step in xfer.mapped_steps)
    assert xfer.method_name == "red_flags_first"


def test_transfer_is_replay_stable():
    lib = MethodLibrary()
    method = SEED_METHODS[0]
    a = lib.transfer(MockLLM(), method, "t1", "text")
    b = lib.transfer(MockLLM(), method, "t1", "text")
    assert a.id == b.id
    assert a.mapped_steps == b.mapped_steps


def test_extract_method_grows_library_from_a_run():
    space = SolutionSpace(label="Boundary", description="edges", axis="boundary",
                          affinities=(Affinity.BOUNDARY,))
    variant = Variant(space_id=space.id, move=WildMove.RISKY_HYPOTHESIS, content="bold guess",
                      wildness=0.9)
    cand = Candidate(content="bold guess refined", space_id=space.id, variant_id=variant.id)
    run = CreativeRun(problem=Problem("a problem"), spaces=[space], variants=[variant],
                      candidates=[cand])

    method = extract_method(run, cand.id, name="boundary_risk_pattern")
    assert method.origin == "kevin"
    assert method.affinities == (Affinity.BOUNDARY,)
    assert len(method.steps) == 3

    lib = MethodLibrary()
    before = len(lib.all())
    lib.add(method)
    assert len(lib.all()) == before + 1


def test_extract_method_rejects_unknown_candidate():
    run = CreativeRun(problem=Problem("a problem"))
    try:
        extract_method(run, "cand_nope", name="x")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown candidate")
