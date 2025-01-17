# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1PriorityClassType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


from .v1_object_meta import V1ObjectMetaType


@dataclass
class V1PriorityClassType:
    api_version: str
    description: str
    global_default: bool
    kind: str
    metadata: V1ObjectMetaType
    preemption_policy: str
    value: int
