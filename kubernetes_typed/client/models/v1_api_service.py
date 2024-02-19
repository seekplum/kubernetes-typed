# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1APIServiceType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


from .v1_object_meta import V1ObjectMetaType
from .v1_api_service_spec import V1APIServiceSpecType
from .v1_api_service_status import V1APIServiceStatusType


@dataclass
class V1APIServiceType:
    api_version: str
    kind: str
    metadata: V1ObjectMetaType
    spec: V1APIServiceSpecType
    status: V1APIServiceStatusType
