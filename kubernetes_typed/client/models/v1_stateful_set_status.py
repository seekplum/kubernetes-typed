# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1StatefulSetStatusType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List

from .v1_stateful_set_condition import V1StatefulSetConditionType


@dataclass
class V1StatefulSetStatusType:
    available_replicas: int
    collision_count: int
    conditions: List[V1StatefulSetConditionType]
    current_replicas: int
    current_revision: str
    observed_generation: int
    ready_replicas: int
    replicas: int
    update_revision: str
    updated_replicas: int
