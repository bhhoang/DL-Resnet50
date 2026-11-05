"""Typed experiment configuration loaded from YAML, with ``key.path=value`` overrides."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

Arch = Literal["resnet50", "resnet50_transfer"]
OptimizerName = Literal["sgd", "adam", "adamw"]
ScheduleName = Literal["step", "cosine", "constant"]


@dataclass
class AugmentationConfig:
    horizontal_flip: bool = True
    rotation_degrees: float = 0.0  # max |angle|; 0 disables
    translation: float = 0.0  # fraction of height/width; 0 disables
    zoom: float = 0.0  # fraction; 0 disables
    contrast: float = 0.0  # factor; 0 disables


@dataclass
class DataConfig:
    data_dir: str = "data"  # contains <train_split>/, <val_split>/, optional <test_split>/
    train_split: str = "train"
    val_split: str = "val"
    test_split: str | None = None
    image_size: tuple[int, int] = (224, 224)
    batch_size: int = 32
    seed: int = 42
    cache: bool = False
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass
class ModelConfig:
    arch: Arch = "resnet50"
    weights: str | None = "imagenet"  # only used by resnet50_transfer
    zero_init_residual: bool = True  # gamma=0 on last BN of each block (Goyal et al., 2017)
    stride_in_3x3: bool = False  # False = original v1 (stride on first 1x1); True = "v1.5"
    dropout: float = 0.0
    freeze_backbone: bool = True  # only used by resnet50_transfer


@dataclass
class TrainConfig:
    epochs: int = 25
    optimizer: OptimizerName = "sgd"
    learning_rate: float = 0.1
    momentum: float = 0.9
    nesterov: bool = False
    weight_decay: float = 1e-4
    schedule: ScheduleName = "step"
    drop_every_epochs: int = 7
    drop_factor: float = 0.1
    warmup_epochs: int = 0
    # Transfer learning phase 2: unfreeze the top `fine_tune_layers` backbone layers.
    fine_tune_epochs: int = 0
    fine_tune_layers: int = 0
    fine_tune_learning_rate: float = 1e-4
    early_stopping_patience: int | None = None
    mixed_precision: bool = False
    distribute: bool = False  # tf.distribute.MirroredStrategy over all visible GPUs
    output_dir: str = "runs"
    run_name: str = "experiment"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.train.output_dir) / self.train.run_name

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["data"]["image_size"] = list(self.data.image_size)
        return d

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def _from_dict(cls: type, data: dict[str, Any] | None) -> Any:
    data = dict(data or {})
    hints = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(hints)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        default = getattr(cls(), name)
        if dataclasses.is_dataclass(default):
            kwargs[name] = _from_dict(type(default), value)
        elif name == "image_size":
            if isinstance(value, int):
                value = (value, value)
            kwargs[name] = tuple(int(v) for v in value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _set_path(d: dict[str, Any], dotted: str, raw: str) -> None:
    keys = dotted.split(".")
    node = d
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = yaml.safe_load(raw)


def validate(cfg: ExperimentConfig) -> ExperimentConfig:
    if cfg.model.arch not in ("resnet50", "resnet50_transfer"):
        raise ValueError(f"model.arch must be resnet50|resnet50_transfer, got {cfg.model.arch!r}")
    if cfg.train.optimizer not in ("sgd", "adam", "adamw"):
        raise ValueError(f"train.optimizer must be sgd|adam|adamw, got {cfg.train.optimizer!r}")
    if cfg.train.schedule not in ("step", "cosine", "constant"):
        raise ValueError(f"train.schedule must be step|cosine|constant, got {cfg.train.schedule!r}")
    if cfg.train.epochs < 1 or cfg.data.batch_size < 1:
        raise ValueError("train.epochs and data.batch_size must be >= 1")
    if not 0.0 <= cfg.model.dropout < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")
    if cfg.train.fine_tune_epochs and cfg.model.arch != "resnet50_transfer":
        raise ValueError("train.fine_tune_epochs > 0 requires model.arch=resnet50_transfer")
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> ExperimentConfig:
    """Load YAML (optional) and apply ``section.key=value`` overrides (values parsed as YAML)."""
    raw: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text()) or {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.path=value, got {item!r}")
        key, value = item.split("=", 1)
        _set_path(raw, key.strip(), value)
    return validate(_from_dict(ExperimentConfig, raw))
