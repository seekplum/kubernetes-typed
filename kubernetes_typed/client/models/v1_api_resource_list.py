# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1APIResourceListType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List

from .v1_api_resource import V1APIResourceType


@dataclass
class V1APIResourceListType:
    api_version: str
    group_version: str
    kind: str
    resources: List[V1APIResourceType]
