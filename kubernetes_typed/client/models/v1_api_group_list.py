# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1APIGroupListType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List

from .v1_api_group import V1APIGroupType


@dataclass
class V1APIGroupListType:
    api_version: str
    groups: List[V1APIGroupType]
    kind: str
