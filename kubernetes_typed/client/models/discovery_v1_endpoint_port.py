# Code generated by `typeddictgen`. DO NOT EDIT.
"""DiscoveryV1EndpointPortType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


@dataclass
class DiscoveryV1EndpointPortType:
    app_protocol: str
    name: str
    port: int
    protocol: str