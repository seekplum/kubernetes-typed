# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1VolumeNodeAffinityType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


from .v1_node_selector import V1NodeSelectorType


@dataclass
class V1VolumeNodeAffinityType:
    required: V1NodeSelectorType
