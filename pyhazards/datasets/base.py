from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class FeatureSpec:
    """Describes input features (shapes, dtypes, normalization)."""
    input_dim: Optional[int] = None
    channels: Optional[int] = None
    description: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LabelSpec:
    """Describes labels/targets for downstream tasks."""
    num_targets: Optional[int] = None
    task_type: str = "regression"  # classification|regression|segmentation
    description: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataSplit:
    """Container for a single split."""
    inputs: Any
    targets: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataBundle:
    """
    Bundle of train/val/test splits plus metadata.
    Keeps feature/label specs to make model construction easy.
    """
    splits: Dict[str, DataSplit]
    feature_spec: FeatureSpec
    label_spec: LabelSpec
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_split(self, name: str) -> DataSplit:
        if name not in self.splits:
            raise KeyError(f"Split '{name}' not found. Available: {list(self.splits.keys())}")
        return self.splits[name]


class Transform(Protocol):
    """Callable data transform."""

    def __call__(self, bundle: DataBundle) -> DataBundle:
        ...


class Dataset:
    """
    Base class for hazard datasets.
    Subclasses should load data and return a DataBundle with splits ready for training.
    """

    name: str = "base"

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir

    def load(self, split: Optional[str] = None, transforms: Optional[List[Transform]] = None) -> DataBundle:
        """
        Return a DataBundle. Optionally return a specific split if provided.
        """
        bundle = self._load()
        if transforms:
            for t in transforms:
                bundle = t(bundle)
        if split:
            return DataBundle(
                splits={split: bundle.get_split(split)},
                feature_spec=bundle.feature_spec,
                label_spec=bundle.label_spec,
                metadata=bundle.metadata,
            )
        return bundle

    def _load(self) -> DataBundle:
        raise NotImplementedError("Subclasses must implement _load() to return a DataBundle.")
