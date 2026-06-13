import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kevin.api import app  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "Kevin" in r.text


def test_methods_are_content_free_over_the_wire():
    r = client.get("/api/methods")
    assert r.status_code == 200
    names = {m["name"] for m in r.json()["methods"]}
    assert "claim_splitting" in names
    blob = " ".join(s for m in r.json()["methods"] for s in m["steps"]).lower()
    assert "patient" not in blob and "plaintiff" not in blob


def test_run_returns_routed_possibility_space():
    payload = {
        "statement": "how do we make onboarding feel less like paperwork?",
        "domain": "people-ops",
        "constraints": ["no extra headcount"],
        "known_approaches": ["checklists", "a welcome email"],
        "top_spaces": 2,
    }
    r = client.post("/api/run", json=payload)
    assert r.status_code == 200
    d = r.json()
    assert d["stats"]["routed"] >= 1
    assert d["stats"]["variants"] > 0
    assert any(s["routed"] for s in d["spaces"])
    # every listed candidate falls in exactly one verdict bucket
    total = len(d["promising"]) + len(d["tentative"]) + len(d["rejected"])
    assert total == d["stats"]["candidates"]
    assert set(d["decision_axes"]) == {"direction", "taste", "risk", "value"}


def test_run_rejects_empty_statement():
    r = client.post("/api/run", json={"statement": "x"})
    assert r.status_code == 422  # min_length guard


def test_run_is_replay_stable_over_the_wire():
    payload = {"statement": "design a calmer morning routine", "top_spaces": 2}
    a = client.post("/api/run", json=payload).json()
    b = client.post("/api/run", json=payload).json()
    assert [c["score"] for c in a["promising"]] == [c["score"] for c in b["promising"]]


def _a_candidate_id(run: dict) -> str:
    for bucket in ("promising", "tentative", "rejected"):
        if run[bucket]:
            return run[bucket][0]["id"]
    raise AssertionError("run produced no candidates")


def test_promote_grows_the_library_and_is_idempotent():
    run = client.post("/api/run", json={"statement": "grow a habit that actually sticks"}).json()
    before = run["library"]["learned"]
    cand_id = _a_candidate_id(run)

    r1 = client.post("/api/promote", json={"run_id": run["run_id"], "candidate_id": cand_id,
                                           "name": "habit_loop_pattern"})
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["already_known"] is False
    assert d1["learned"]["learned"] is True
    assert d1["library"]["learned"] == before + 1

    # Re-promoting the same candidate under the same name is a no-op (idempotent).
    r2 = client.post("/api/promote", json={"run_id": run["run_id"], "candidate_id": cand_id,
                                           "name": "habit_loop_pattern"})
    assert r2.json()["already_known"] is True
    assert r2.json()["library"]["learned"] == before + 1


def test_promoted_method_appears_in_the_library_endpoint():
    run = client.post("/api/run", json={"statement": "make standups worth attending"}).json()
    cand_id = _a_candidate_id(run)
    client.post("/api/promote", json={"run_id": run["run_id"], "candidate_id": cand_id,
                                      "name": "standup_value_pattern"})
    methods = client.get("/api/methods").json()["methods"]
    learned = [m for m in methods if m["learned"]]
    assert any(m["name"] == "standup_value_pattern" for m in learned)
    assert all(m["origin"] == "kevin" for m in learned)


def test_promote_unknown_run_is_404():
    r = client.post("/api/promote", json={"run_id": "run_does_not_exist", "candidate_id": "cand_x"})
    assert r.status_code == 404


def test_promote_unknown_candidate_is_404():
    run = client.post("/api/run", json={"statement": "a fresh problem to route"}).json()
    r = client.post("/api/promote", json={"run_id": run["run_id"], "candidate_id": "cand_nope"})
    assert r.status_code == 404
