# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1PreferredSchedulingTermType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


from .v1_node_selector_term import V1NodeSelectorTermType


@dataclass
class V1PreferredSchedulingTermType:
    preference: V1NodeSelectorTermType
    weight: int
