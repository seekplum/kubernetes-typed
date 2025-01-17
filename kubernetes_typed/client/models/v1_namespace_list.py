# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1NamespaceListType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List

from .v1_namespace import V1NamespaceType
from .v1_list_meta import V1ListMetaType


@dataclass
class V1NamespaceListType:
    api_version: str
    items: List[V1NamespaceType]
    kind: str
    metadata: V1ListMetaType
