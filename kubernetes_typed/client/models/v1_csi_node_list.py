# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1CSINodeListType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List

from .v1_csi_node import V1CSINodeType
from .v1_list_meta import V1ListMetaType


@dataclass
class V1CSINodeListType:
    api_version: str
    items: List[V1CSINodeType]
    kind: str
    metadata: V1ListMetaType
