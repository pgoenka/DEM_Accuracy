from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineContext:

    aoi: Any

    raw_dems: dict = field(default_factory=dict)

    lidar: dict = field(default_factory=dict)

    satellite: dict = field(default_factory=dict)

    aligned_dems: dict = field(default_factory=dict)

    features: dict = field(default_factory=dict)

    outputs: dict = field(default_factory=dict)