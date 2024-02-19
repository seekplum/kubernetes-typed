# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1EndpointsListType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List

from .v1_endpoints import V1EndpointsType
from .v1_list_meta import V1ListMetaType


@dataclass
class V1EndpointsListType:
    api_version: str
    items: List[V1EndpointsType]
    kind: str
    metadata: V1ListMetaType
