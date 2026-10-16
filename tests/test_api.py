"""API contract tests (the model is loaded once per session)."""

import pytest
from fastapi.testclient import TestClient

from server.app import app

PROMPT = "The Colosseum is located in the city of"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_model_info(client):
    info = client.get("/api/model").json()
    assert info["n_layers"] == 12 and "linear" in info["twin_levels"]


def test_run_shapes(client):
    r = client.post("/api/run", json={"prompt": PROMPT}).json()
    n = len(r["tokens"])
    assert r["target"]["token"] == " Rome"
    assert len(r["attention"]) == 12 and len(r["attention"][0][0]) == n
    assert len(r["logit_lens"]["top_token"]) == 13
    assert abs(sum(t["prob"] for t in r["top"])) <= 1.0001


def test_intervene_changes_distribution(client):
    body = {"prompt": PROMPT, "interventions": [{"kind": "head", "layer": 0, "head": 1}]}
    r = client.post("/api/intervene", json=body).json()
    assert r["kl"] >= 0 and r["before"][0]["token"] == " Rome"


def test_explain_and_whatif(client):
    e = client.post("/api/twin/explain", json={"prompt": PROMPT}).json()
    assert e["features"], "expected features in the reasoning path"
    f = e["features"][0]
    w = client.post(
        "/api/twin/whatif",
        json={
            "prompt": PROMPT,
            "layer": f["layer"],
            "pos": f["pos"],
            "feature": f["feature"],
            "value": 0.0,
        },
    ).json()
    assert set(w["levels"]) >= {"direct", "linear", "replacement_full", "real"}
    for level in w["levels"].values():
        assert level["kl"] >= -1e-6 and abs(level["dlogp"]) < 1e3
    # Ablating the strongest positive feature must lower the prediction in the direct twin.
    if f["effect"] > 0:
        assert w["levels"]["direct"]["dlogp"] < 0
    assert w["trust"] is None or 0.0 <= w["trust"] <= 1.0


def test_rejects_bad_input(client):
    assert client.post("/api/run", json={"prompt": ""}).status_code == 422
    assert client.post("/api/run", json={"prompt": "word " * 200}).status_code == 400
    bad = {"prompt": PROMPT, "layer": 99, "pos": 0, "feature": 0}
    assert client.post("/api/twin/whatif", json=bad).status_code == 422
    assert client.get("/api/case-studies/nope").status_code == 404
