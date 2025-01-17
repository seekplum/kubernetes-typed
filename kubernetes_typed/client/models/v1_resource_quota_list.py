# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1ResourceQuotaListType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List

from .v1_resource_quota import V1ResourceQuotaType
from .v1_list_meta import V1ListMetaType


@dataclass
class V1ResourceQuotaListType:
    api_version: str
    items: List[V1ResourceQuotaType]
    kind: str
    metadata: V1ListMetaType
