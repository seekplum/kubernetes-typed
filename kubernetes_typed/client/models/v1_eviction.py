# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1EvictionType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


from .v1_delete_options import V1DeleteOptionsType
from .v1_object_meta import V1ObjectMetaType


@dataclass
class V1EvictionType:
    api_version: str
    delete_options: V1DeleteOptionsType
    kind: str
    metadata: V1ObjectMetaType
