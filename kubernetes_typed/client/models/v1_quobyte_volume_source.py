# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1QuobyteVolumeSourceType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


@dataclass
class V1QuobyteVolumeSourceType:
    group: str
    read_only: bool
    registry: str
    tenant: str
    user: str
    volume: str
