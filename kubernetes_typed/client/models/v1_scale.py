# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1ScaleType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


from .v1_object_meta import V1ObjectMetaType
from .v1_scale_spec import V1ScaleSpecType
from .v1_scale_status import V1ScaleStatusType


@dataclass
class V1ScaleType:
    api_version: str
    kind: str
    metadata: V1ObjectMetaType
    spec: V1ScaleSpecType
    status: V1ScaleStatusType
