# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1NodeStatusType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import Dict, List

from .v1_node_address import V1NodeAddressType
from .v1_node_condition import V1NodeConditionType
from .v1_node_config_status import V1NodeConfigStatusType
from .v1_node_daemon_endpoints import V1NodeDaemonEndpointsType
from .v1_container_image import V1ContainerImageType
from .v1_node_system_info import V1NodeSystemInfoType
from .v1_attached_volume import V1AttachedVolumeType


@dataclass
class V1NodeStatusType:
    addresses: List[V1NodeAddressType]
    allocatable: Dict[str, str]
    capacity: Dict[str, str]
    conditions: List[V1NodeConditionType]
    config: V1NodeConfigStatusType
    daemon_endpoints: V1NodeDaemonEndpointsType
    images: List[V1ContainerImageType]
    node_info: V1NodeSystemInfoType
    phase: str
    volumes_attached: List[V1AttachedVolumeType]
    volumes_in_use: List[str]
