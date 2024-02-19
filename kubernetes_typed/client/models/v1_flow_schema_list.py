# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1FlowSchemaListType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List

from .v1_flow_schema import V1FlowSchemaType
from .v1_list_meta import V1ListMetaType


@dataclass
class V1FlowSchemaListType:
    api_version: str
    items: List[V1FlowSchemaType]
    kind: str
    metadata: V1ListMetaType
