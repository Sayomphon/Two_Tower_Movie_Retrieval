"""Configuration dataclasses + JSON persistence.

Every design decision that affects results (split rule, positive-interaction rule,
hyperparameters) is declared here and serialized to artifacts/split_config.json
so that runs stay reproducible and auditable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

MODEL_VERSION = "ml100k-retrieval-v1"


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from cwd until pyproject.toml is found (supports running from notebooks/)."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return current


@dataclass(frozen=True)
class Paths:
    """Every file location in the project — data/ and artifacts/ are gitignored"""

    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def index_dir(self) -> Path:
        return self.artifacts_dir / "bruteforce_index"

    @property
    def model_dir(self) -> Path:
        return self.artifacts_dir / "retrieval_model"

    @classmethod
    def default(cls) -> Paths:
        return cls(root=find_project_root())


@dataclass(frozen=True)
class SplitConfig:
    """Per-user temporal leave-last-k-out split.

    - test  = each user's most recent interactions (k_test of them)
    - val   = the interactions just before test (k_val of them)
    - train = everything else
    Users with history < min_history are dropped entirely (declared policy)
    """

    k_test: int = 1
    k_val: int = 1
    min_train: int = 3
    # positive-interaction rule: None = every rating counts as an interaction,
    # or set a threshold, e.g. 4.0 = only rating >= 4 counts as positive
    positive_rating_threshold: float | None = None
    seed: int = 42

    @property
    def min_history(self) -> int:
        return self.k_test + self.k_val + self.min_train


@dataclass(frozen=True)
class ModelConfig:
    """Two-tower hyperparameters — defaults are the main run (R1) in the experiment matrix."""

    embedding_dim: int = 32
    learning_rate: float = 0.1  # Adagrad — standard for sparse embedding retrieval
    epochs: int = 15
    batch_size: int = 256
    l2_regularization: float = 0.0
    seed: int = 42


@dataclass
class ExperimentConfig:
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    model_version: str = MODEL_VERSION

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_version": self.model_version,
            "split": asdict(self.split),
            "model": asdict(self.model),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    @classmethod
    def from_json(cls, path: Path) -> ExperimentConfig:
        payload = json.loads(path.read_text())
        return cls(
            split=SplitConfig(**payload["split"]),
            model=ModelConfig(**payload["model"]),
            model_version=payload.get("model_version", MODEL_VERSION),
        )
