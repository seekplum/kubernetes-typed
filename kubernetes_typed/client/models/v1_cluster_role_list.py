# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1ClusterRoleListType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List

from .v1_cluster_role import V1ClusterRoleType
from .v1_list_meta import V1ListMetaType


@dataclass
class V1ClusterRoleListType:
    api_version: str
    items: List[V1ClusterRoleType]
    kind: str
    metadata: V1ListMetaType
