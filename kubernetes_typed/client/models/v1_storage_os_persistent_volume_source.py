# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1StorageOSPersistentVolumeSourceType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


from .v1_object_reference import V1ObjectReferenceType


@dataclass
class V1StorageOSPersistentVolumeSourceType:
    fs_type: str
    read_only: bool
    secret_ref: V1ObjectReferenceType
    volume_name: str
    volume_namespace: str
