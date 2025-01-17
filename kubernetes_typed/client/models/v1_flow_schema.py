# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1FlowSchemaType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


from .v1_object_meta import V1ObjectMetaType
from .v1_flow_schema_spec import V1FlowSchemaSpecType
from .v1_flow_schema_status import V1FlowSchemaStatusType


@dataclass
class V1FlowSchemaType:
    api_version: str
    kind: str
    metadata: V1ObjectMetaType
    spec: V1FlowSchemaSpecType
    status: V1FlowSchemaStatusType
