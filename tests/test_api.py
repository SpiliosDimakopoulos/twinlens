"""API contract tests that need the model (loaded once per module).

Model-free validation tests live in test_api_validation.py.
"""

import pytest
from fastapi.testclient import TestClient

from server import app as api
from server.app import app

PROMPT = "The Colosseum is located in the city of"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_model_info(client):
    info = client.get("/api/model").json()
    assert info["n_layers"] == 12
    assert info["d_sae"] == api.D_SAE
    assert "linear" in info["twin_levels"]


def test_run_shapes(client):
    r = client.post("/api/run", json={"prompt": PROMPT}).json()
    n = len(r["tokens"])
    assert r["target"]["token"] == " Rome"
    assert len(r["attention"]) == 12
    assert len(r["attention"][0][0]) == n
    assert len(r["logit_lens"]["top_token"]) == 13
    assert sum(t["prob"] for t in r["top"]) <= 1.0001


def test_intervene_changes_distribution(client):
    body = {"prompt": PROMPT, "interventions": [{"kind": "head", "layer": 0, "head": 1}]}
    r = client.post("/api/intervene", json=body).json()
    assert r["kl"] > 0
    assert r["before"][0]["token"] == " Rome"


def test_explain_and_whatif(client):
    e = client.post("/api/twin/explain", json={"prompt": PROMPT}).json()
    assert e["features"], "expected features in the reasoning path"
    positive = [f for f in e["features"] if f["effect"] > 0]
    assert positive, "expected a feature that supports the prediction"
    f = max(positive, key=lambda f: f["effect"])
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
        assert level["kl"] >= -1e-6
        assert abs(level["dlogp"]) < 1e3
    # Ablating the strongest positive feature must lower the prediction in the direct twin.
    assert w["levels"]["direct"]["dlogp"] < 0
    assert w["trust"] is None or 0.0 <= w["trust"] <= 1.0
    assert len(w["trust_factors"]) <= 3
    assert all({"signal", "contribution"} <= set(tf) for tf in w["trust_factors"])


def test_rejects_bad_input(client):
    assert client.post("/api/run", json={"prompt": ""}).status_code == 422
    assert client.post("/api/run", json={"prompt": "word " * 200}).status_code == 400
    bad = {"prompt": PROMPT, "layer": 99, "pos": 0, "feature": 0}
    assert client.post("/api/twin/whatif", json=bad).status_code == 422
    assert client.get("/api/case-studies/nope").status_code == 404


@pytest.mark.parametrize(
    "bad",
    [
        {"kind": "head", "layer": 40, "head": 0},
        {"kind": "head", "layer": 0, "head": 99},
        {"kind": "neuron", "layer": 0, "neuron": 10**6},
        {"kind": "head", "layer": 0, "head": 0, "positions": [500]},
        {"kind": "head", "layer": 0, "head": 0, "positions": []},
        {"kind": "head", "layer": 0, "head": 0, "mode": "mean"},
        {"kind": "direction", "hook": "blocks.0.hook_resid_mid", "direction": [1.0], "delta": 1.0},
    ],
    ids=["layer", "head", "neuron", "position", "empty_positions", "mean", "direction"],
)
def test_intervene_rejects_bad_edit(client, bad):
    """Bad edits must be a 400, never an IndexError inside a hook (a 500)."""
    r = client.post("/api/intervene", json={"prompt": PROMPT, "interventions": [bad]})
    assert r.status_code == 400


def test_intervene_rejects_too_many(client):
    one = {"kind": "head", "layer": 0, "head": 0}
    body = {"prompt": PROMPT, "interventions": [one] * (api.MAX_INTERVENTIONS + 1)}
    assert client.post("/api/intervene", json=body).status_code == 422


def test_multi_token_target_is_400(client):
    r = client.post("/api/run", json={"prompt": PROMPT, "target": " Rome Paris"})
    assert r.status_code == 400


def _collect(ws):
    frames = []
    while True:
        frame = ws.receive_json()
        frames.append(frame)
        if "done" in frame or "error" in frame:
            return frames


def test_websocket_generates_tokens(client):
    with client.websocket_connect("/ws/generate") as ws:
        ws.send_json({"prompt": PROMPT, "max_new_tokens": 2})
        frames = _collect(ws)
    assert frames[-1] == {"done": True}
    tokens = frames[:-1]
    assert len(tokens) == 2
    assert all({"token", "top", "lens", "lens_prob"} <= set(t) for t in tokens)


def test_websocket_overlong_prompt_gives_error_frame(client):
    with client.websocket_connect("/ws/generate") as ws:
        ws.send_json({"prompt": "word " * 200, "max_new_tokens": 2})
        frames = _collect(ws)
    assert len(frames) == 1
    assert "error" in frames[0]


def test_websocket_clamps_max_new_tokens(client, monkeypatch):
    monkeypatch.setattr(api, "GEN_EXTRA", 3)
    with client.websocket_connect("/ws/generate") as ws:
        ws.send_json({"prompt": PROMPT, "max_new_tokens": 1000})
        frames = _collect(ws)
    assert len(frames) == 3 + 1
    assert frames[-1] == {"done": True}
