# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1DownwardAPIVolumeSourceType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List

from .v1_downward_api_volume_file import V1DownwardAPIVolumeFileType


@dataclass
class V1DownwardAPIVolumeSourceType:
    default_mode: int
    items: List[V1DownwardAPIVolumeFileType]
