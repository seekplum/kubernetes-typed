# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1ConfigMapVolumeSourceType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List

from .v1_key_to_path import V1KeyToPathType


@dataclass
class V1ConfigMapVolumeSourceType:
    default_mode: int
    items: List[V1KeyToPathType]
    name: str
    optional: bool
