# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1NodeSelectorTermType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List

from .v1_node_selector_requirement import V1NodeSelectorRequirementType


@dataclass
class V1NodeSelectorTermType:
    match_expressions: List[V1NodeSelectorRequirementType]
    match_fields: List[V1NodeSelectorRequirementType]
