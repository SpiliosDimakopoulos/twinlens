"""API input validation that needs no model weights (runs without downloading GPT-2)."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from server import app as api
from twinlens.causal.interventions import DirectionEdit, HeadAblation, NeuronEdit

CFG = SimpleNamespace(n_layers=12, n_heads=12, d_mlp=3072)


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)  # no lifespan, no model


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/api/run", {"prompt": ""}),
        ("/api/run", {"prompt": "x" * 5000}),
        ("/api/twin/whatif", {"prompt": "hi", "layer": 99, "pos": 0, "feature": 0}),
        ("/api/twin/whatif", {"prompt": "hi", "layer": 3, "pos": 0, "feature": 10**6}),
        ("/api/intervene", {"prompt": "hi", "interventions": []}),
        ("/api/intervene", {"prompt": "hi", "interventions": [{"kind": "bogus"}]}),
        ("/api/patch", {"clean": "a", "corrupt": "", "correct": "b", "wrong": "c"}),
    ],
)
def test_schema_rejects(client, path, body):
    assert client.post(path, json=body).status_code == 422


def test_unknown_case_study_is_404(client):
    assert client.get("/api/case-studies/nope").status_code == 404


@pytest.mark.parametrize(
    "item",
    [
        HeadAblation(layer=12, head=0),
        HeadAblation(layer=-1, head=0),
        HeadAblation(layer=0, head=12),
        HeadAblation(layer=0, head=0, mode="mean"),
        HeadAblation(layer=0, head=0, positions=[3]),
        HeadAblation(layer=0, head=0, positions=[]),
        NeuronEdit(layer=0, neuron=3072, value=0.0),
        DirectionEdit(hook="x", direction=[1.0], delta=1.0),
    ],
    ids=[
        "layer_high",
        "layer_negative",
        "head",
        "mean_ablation",
        "position",
        "empty_positions",
        "neuron",
        "direction_edit",
    ],
)
def test_check_interventions_rejects(item):
    with pytest.raises(HTTPException) as e:
        api._check_interventions(CFG, [item], n_pos=3)
    assert e.value.status_code == 400


def test_check_interventions_too_many():
    items = [HeadAblation(layer=0, head=0)] * (api.MAX_INTERVENTIONS + 1)
    with pytest.raises(HTTPException) as e:
        api._check_interventions(CFG, items, n_pos=3)
    assert e.value.status_code == 400


def test_check_interventions_accepts_valid():
    items = [HeadAblation(layer=11, head=11, positions=[0, 2]), NeuronEdit(layer=0, neuron=3071)]
    api._check_interventions(CFG, items, n_pos=3)


def test_single_token_error_is_400_but_other_errors_propagate():
    class Fake:
        def to_single_token(self, s):
            if s == "bug":
                raise RuntimeError("internal")
            raise ValueError(f"{s!r} is 2 tokens")

    with pytest.raises(HTTPException) as e:
        api._single_token(Fake(), "ab")
    assert e.value.status_code == 400
    with pytest.raises(RuntimeError):
        api._single_token(Fake(), "bug")


def test_internal_value_error_is_not_masked_as_400():
    assert not any(exc is ValueError for exc in api.app.exception_handlers)


@pytest.mark.parametrize(
    "msg", [{"nope": 1}, [], {"prompt": ""}, {"prompt": "a", "max_new_tokens": "3"}]
)
def test_websocket_rejects_malformed_message(client, msg):
    with client.websocket_connect("/ws/generate") as ws:
        ws.send_json(msg)
        assert "error" in ws.receive_json()


def test_websocket_rejects_foreign_origin(client):
    headers = {"origin": "http://evil.example"}
    with (
        pytest.raises(WebSocketDisconnect) as e,
        client.websocket_connect("/ws/generate", headers=headers),
    ):
        pass
    assert e.value.code == 1008


def test_origin_lists_share_the_dev_origin():
    assert set(api.CORS_ORIGINS) <= api.WS_ORIGINS


def test_kl_and_dlogp():
    import torch

    p = torch.tensor([0.5, 0.5]).log()
    q = torch.tensor([0.25, 0.75]).log()
    assert api._kl(p, p) == pytest.approx(0.0, abs=1e-7)
    assert api._kl(p, q) > 0
    assert api._dlogp(p, q, 1) == pytest.approx(float(q[1] - p[1]))
