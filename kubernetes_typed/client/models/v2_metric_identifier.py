# Code generated by `typeddictgen`. DO NOT EDIT.
"""V2MetricIdentifierType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


from .v1_label_selector import V1LabelSelectorType


@dataclass
class V2MetricIdentifierType:
    name: str
    selector: V1LabelSelectorType
