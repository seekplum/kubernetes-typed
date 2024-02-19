# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1StorageClassType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import Dict, List

from .v1_topology_selector_term import V1TopologySelectorTermType
from .v1_object_meta import V1ObjectMetaType


@dataclass
class V1StorageClassType:
    allow_volume_expansion: bool
    allowed_topologies: List[V1TopologySelectorTermType]
    api_version: str
    kind: str
    metadata: V1ObjectMetaType
    mount_options: List[str]
    parameters: Dict[str, str]
    provisioner: str
    reclaim_policy: str
    volume_binding_mode: str
