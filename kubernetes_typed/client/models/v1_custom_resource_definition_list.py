# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1CustomResourceDefinitionListType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List

from .v1_custom_resource_definition import V1CustomResourceDefinitionType
from .v1_list_meta import V1ListMetaType


@dataclass
class V1CustomResourceDefinitionListType:
    api_version: str
    items: List[V1CustomResourceDefinitionType]
    kind: str
    metadata: V1ListMetaType
