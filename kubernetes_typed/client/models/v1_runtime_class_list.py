# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1RuntimeClassListType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List

from .v1_runtime_class import V1RuntimeClassType
from .v1_list_meta import V1ListMetaType


@dataclass
class V1RuntimeClassListType:
    api_version: str
    items: List[V1RuntimeClassType]
    kind: str
    metadata: V1ListMetaType
