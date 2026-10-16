"""Trust model: predicts, from reliability signals alone, whether the twin's prediction holds.

A standardised logistic regression trained on the fidelity sweep. It is deliberately simple:
coefficients are inspectable and it ships as a small JSON file with no ML dependency.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


def transform(name: str, value: float) -> float:
    """Heavy-tailed signals are log-scaled before standardisation."""
    if name in {"activation", "rel_edit_size", "abs_direct_effect", "n_active_downstream"}:
        return math.log1p(abs(value))
    return value


@dataclass
class TrustModel:
    features: list[str]
    mean: list[float]
    std: list[float]
    coef: list[float]
    intercept: float
    target: str  # description of what "trustworthy" means
    metrics: dict[str, float]

    def _x(self, signals: dict[str, float]) -> torch.Tensor:
        raw = torch.tensor([transform(f, float(signals[f])) for f in self.features])
        return (raw - torch.tensor(self.mean)) / torch.tensor(self.std)

    def predict(self, signals: dict[str, float]) -> float:
        z = float(self._x(signals) @ torch.tensor(self.coef)) + self.intercept
        return 1.0 / (1.0 + math.exp(-z))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> TrustModel:
        return cls(**json.loads(Path(path).read_text()))


def fit_logistic(
    X: torch.Tensor, y: torch.Tensor, l2: float = 1e-2, steps: int = 500
) -> tuple[torch.Tensor, float]:
    """L2-regularised logistic regression fitted with L-BFGS. X must be standardised."""
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y) + l2 * w.pow(2).sum()
        )
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    return w.detach(), float(b.detach())


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Probability that a random positive is scored above a random negative (ties count 1/2)."""
    pos, neg = scores[labels > 0.5], scores[labels <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).float() + 0.5 * (diff == 0).float()).mean())
