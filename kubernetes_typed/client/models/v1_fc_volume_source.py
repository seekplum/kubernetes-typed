# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1FCVolumeSourceType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List


@dataclass
class V1FCVolumeSourceType:
    fs_type: str
    lun: int
    read_only: bool
    target_ww_ns: List[str]
    wwids: List[str]
