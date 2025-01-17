# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1JobType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


from .v1_object_meta import V1ObjectMetaType
from .v1_job_spec import V1JobSpecType
from .v1_job_status import V1JobStatusType


@dataclass
class V1JobType:
    api_version: str
    kind: str
    metadata: V1ObjectMetaType
    spec: V1JobSpecType
    status: V1JobStatusType
