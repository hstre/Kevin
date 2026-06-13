from kevin import Kevin, Problem
from kevin.layer9 import Layer9Store
from kevin.method_library import SEED_METHODS, MethodLibrary, extract_method


def _store(tmp_path):
    return Layer9Store(tmp_path / "layer9.jsonl")


def _learned_method():
    run = Kevin().run(Problem("how do we make onboarding feel less like paperwork?"))
    method = extract_method(run, run.candidates[0].id)
    return run, method


def test_append_and_reload_roundtrip(tmp_path):
    store = _store(tmp_path)
    assert store.load_methods() == []
    _, method = _learned_method()
    store.append(method, run_id="run_x", candidate_id="cand_y")

    reloaded = store.load_methods()
    assert len(reloaded) == 1
    assert reloaded[0].id == method.id
    assert reloaded[0].origin == "kevin"
    assert reloaded[0].affinities == method.affinities


def test_ledger_is_append_only_and_dedupes_by_id(tmp_path):
    store = _store(tmp_path)
    _, method = _learned_method()
    store.append(method, run_id="r1", candidate_id="c1")
    store.append(method, run_id="r2", candidate_id="c2")  # same shape -> same id
    # Two physical lines (append-only, never rewritten)...
    assert store.path.read_text().count("\n") == 2
    # ...but load de-duplicates by content-hash id.
    assert len(store.load_methods()) == 1


def test_extract_method_is_idempotent_by_shape():
    """Auto-named extraction of the same (axis, move) shape yields the same id."""
    run = Kevin().run(Problem("design a calmer morning routine"))
    cand = run.candidates[0]
    a = extract_method(run, cand.id)
    b = extract_method(run, cand.id)
    assert a.id == b.id
    assert a.name.startswith("learned_")


def test_library_loads_seed_plus_ledger(tmp_path):
    store = _store(tmp_path)
    _, method = _learned_method()
    store.append(method, run_id="r", candidate_id="c")

    lib = MethodLibrary(SEED_METHODS + tuple(store.load_methods()))
    assert len(lib.all()) == len(SEED_METHODS) + 1
    assert any(m.id == method.id for m in lib.all())


def test_learned_method_is_matchable_for_future_runs(tmp_path):
    """A method learned from a run can be transferred in later runs (the loop)."""
    store = _store(tmp_path)
    _, method = _learned_method()
    store.append(method, run_id="r", candidate_id="c")

    lib = MethodLibrary(SEED_METHODS + tuple(store.load_methods()))
    # Its own affinities should retrieve it.
    matched = lib.match(method.affinities, top_k=len(lib.all()))
    assert any(m.id == method.id for m in matched)
