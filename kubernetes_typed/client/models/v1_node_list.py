# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1NodeListType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List

from .v1_node import V1NodeType
from .v1_list_meta import V1ListMetaType


@dataclass
class V1NodeListType:
    api_version: str
    items: List[V1NodeType]
    kind: str
    metadata: V1ListMetaType
