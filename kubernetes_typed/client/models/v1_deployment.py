# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1DeploymentType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


from .v1_object_meta import V1ObjectMetaType
from .v1_deployment_spec import V1DeploymentSpecType
from .v1_deployment_status import V1DeploymentStatusType


@dataclass
class V1DeploymentType:
    api_version: str
    kind: str
    metadata: V1ObjectMetaType
    spec: V1DeploymentSpecType
    status: V1DeploymentStatusType
