# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1DeleteOptionsType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List

from .v1_preconditions import V1PreconditionsType


@dataclass
class V1DeleteOptionsType:
    api_version: str
    dry_run: List[str]
    grace_period_seconds: int
    kind: str
    orphan_dependents: bool
    preconditions: V1PreconditionsType
    propagation_policy: str
