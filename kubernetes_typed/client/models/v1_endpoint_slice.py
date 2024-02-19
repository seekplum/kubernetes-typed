# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1EndpointSliceType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List

from .v1_endpoint import V1EndpointType
from .v1_object_meta import V1ObjectMetaType
from .discovery_v1_endpoint_port import DiscoveryV1EndpointPortType


@dataclass
class V1EndpointSliceType:
    address_type: str
    api_version: str
    endpoints: List[V1EndpointType]
    kind: str
    metadata: V1ObjectMetaType
    ports: List[DiscoveryV1EndpointPortType]
