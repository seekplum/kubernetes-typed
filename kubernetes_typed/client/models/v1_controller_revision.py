# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1ControllerRevisionType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


from .v1_object_meta import V1ObjectMetaType


@dataclass
class V1ControllerRevisionType:
    api_version: str
    data: object
    kind: str
    metadata: V1ObjectMetaType
    revision: int
