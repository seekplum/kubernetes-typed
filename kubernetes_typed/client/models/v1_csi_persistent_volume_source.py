# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1CSIPersistentVolumeSourceType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import Dict

from .v1_secret_reference import V1SecretReferenceType


@dataclass
class V1CSIPersistentVolumeSourceType:
    controller_expand_secret_ref: V1SecretReferenceType
    controller_publish_secret_ref: V1SecretReferenceType
    driver: str
    fs_type: str
    node_expand_secret_ref: V1SecretReferenceType
    node_publish_secret_ref: V1SecretReferenceType
    node_stage_secret_ref: V1SecretReferenceType
    read_only: bool
    volume_attributes: Dict[str, str]
    volume_handle: str
