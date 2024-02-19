# Code generated by `typeddictgen`. DO NOT EDIT.
"""FlowcontrolV1SubjectType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


from .v1_group_subject import V1GroupSubjectType
from .v1_service_account_subject import V1ServiceAccountSubjectType
from .v1_user_subject import V1UserSubjectType


@dataclass
class FlowcontrolV1SubjectType:
    group: V1GroupSubjectType
    kind: str
    service_account: V1ServiceAccountSubjectType
    user: V1UserSubjectType
