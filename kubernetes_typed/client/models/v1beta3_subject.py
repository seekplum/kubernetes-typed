# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1beta3SubjectType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


from .v1beta3_group_subject import V1beta3GroupSubjectType
from .v1beta3_service_account_subject import V1beta3ServiceAccountSubjectType
from .v1beta3_user_subject import V1beta3UserSubjectType


@dataclass
class V1beta3SubjectType:
    group: V1beta3GroupSubjectType
    kind: str
    service_account: V1beta3ServiceAccountSubjectType
    user: V1beta3UserSubjectType
