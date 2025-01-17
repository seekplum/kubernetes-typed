# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1ReplicationControllerType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


from .v1_object_meta import V1ObjectMetaType
from .v1_replication_controller_spec import V1ReplicationControllerSpecType
from .v1_replication_controller_status import V1ReplicationControllerStatusType


@dataclass
class V1ReplicationControllerType:
    api_version: str
    kind: str
    metadata: V1ObjectMetaType
    spec: V1ReplicationControllerSpecType
    status: V1ReplicationControllerStatusType
