# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1VsphereVirtualDiskVolumeSourceType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


@dataclass
class V1VsphereVirtualDiskVolumeSourceType:
    fs_type: str
    storage_policy_id: str
    storage_policy_name: str
    volume_path: str
