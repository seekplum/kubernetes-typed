# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1IngressClassListType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List

from .v1_ingress_class import V1IngressClassType
from .v1_list_meta import V1ListMetaType


@dataclass
class V1IngressClassListType:
    api_version: str
    items: List[V1IngressClassType]
    kind: str
    metadata: V1ListMetaType
