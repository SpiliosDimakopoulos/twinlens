"""CLI pieces that need no model."""

import json

import pytest

from twinlens import cli
from twinlens.cli import AuditReport, Edge, _verdict, main, print_report


@pytest.mark.parametrize(
    ("pred", "real", "want"),
    [
        (-8.12, 0.23, "WRONG SIGN"),
        (-2.03, -0.10, "overstated 20x"),
        (-0.30, -0.10, "overstated 3.0x"),
        (-0.04, -0.20, "understated"),
        (-0.50, -0.45, "ok"),
        (0.001, 0.002, "no effect"),
        # boundaries: exactly 2x and 0.5x are still "ok"; 10x switches to whole numbers
        (-0.20, -0.10, "ok"),
        (-0.05, -0.10, "ok"),
        (1.0, 0.1, "overstated 10x"),
        # a graph that predicts nothing against a large real effect
        (0.0, 1.0, "WRONG SIGN"),
        (0.0, -1.0, "understated"),
        # a large prediction where the real model barely moves: the ratio uses a 1e-3 floor
        (0.5, 0.0, "overstated 500x"),
        (-0.5, 0.005, "overstated 100x"),
    ],
)
def test_verdict(pred, real, want):
    assert _verdict(pred, real) == want


def test_rejects_bad_top():
    with pytest.raises(SystemExit):
        main(["audit", "hello", "--top", "0"])


def _report() -> AuditReport:
    edge = Edge(
        layer=5,
        pos=3,
        feature=1234,
        token=" Rome",
        direct_effect=-0.8,
        graph=-2.0,
        fixed=-0.5,
        gpt2=-0.45,
        graph_verdict="overstated 4.4x",
        fixed_verdict="ok",
        trust=0.25,
    )
    return AuditReport("The Colosseum is in", " Rome", "ablate", [edge])


def test_print_report(capsys):
    print_report(_report())
    out = capsys.readouterr().out
    assert "Prompt: 'The Colosseum is in'" in out
    assert "GPT-2 predicts ' Rome'. Edit: ablate each feature." in out
    row = next(line for line in out.splitlines() if "1234" in line)
    want = "5 Rome 1234 -2.000 -0.500 -0.450 overstated 4.4x ok 25%"
    assert row.split() == want.split()
    assert "off on 1 of 1 edges; with the fix, 0 of 1." in out


def test_report_is_subscriptable_like_the_old_dict():
    report = _report()
    assert report["edges"][0]["graph"] == -2.0


def test_json_output(monkeypatch, capsys):
    calls = []

    def fake_audit(prompt, top, edit):
        calls.append((prompt, top, edit))
        return _report()

    monkeypatch.setattr(cli, "audit", fake_audit)
    assert main(["audit", "The Colosseum is in", "--top", "3", "--json"]) == 0
    assert calls == [("The Colosseum is in", 3, "ablate")]
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert list(data) == ["prompt", "prediction", "edit", "edges"]
    assert list(data["edges"][0]) == [
        "layer",
        "pos",
        "feature",
        "token",
        "direct_effect",
        "graph",
        "fixed",
        "gpt2",
        "graph_verdict",
        "fixed_verdict",
        "trust",
    ]
    assert data["edges"][0]["trust"] == 0.25
    assert "loading GPT-2" in captured.err


def test_halve_warns_about_the_trust_model(monkeypatch, capsys):
    monkeypatch.setattr(cli, "audit", lambda prompt, top, edit: _report())
    main(["audit", "x", "--edit", "halve"])
    assert "trained on ablate and double" in capsys.readouterr().err
