# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1FlexVolumeSourceType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import Dict

from .v1_local_object_reference import V1LocalObjectReferenceType


@dataclass
class V1FlexVolumeSourceType:
    driver: str
    fs_type: str
    options: Dict[str, str]
    read_only: bool
    secret_ref: V1LocalObjectReferenceType
