# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1EndpointType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import Dict, List

from .v1_endpoint_conditions import V1EndpointConditionsType
from .v1_endpoint_hints import V1EndpointHintsType
from .v1_object_reference import V1ObjectReferenceType


@dataclass
class V1EndpointType:
    addresses: List[str]
    conditions: V1EndpointConditionsType
    deprecated_topology: Dict[str, str]
    hints: V1EndpointHintsType
    hostname: str
    node_name: str
    target_ref: V1ObjectReferenceType
    zone: str
