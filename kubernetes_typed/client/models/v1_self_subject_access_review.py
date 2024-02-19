# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1SelfSubjectAccessReviewType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


from .v1_object_meta import V1ObjectMetaType
from .v1_self_subject_access_review_spec import V1SelfSubjectAccessReviewSpecType
from .v1_subject_access_review_status import V1SubjectAccessReviewStatusType


@dataclass
class V1SelfSubjectAccessReviewType:
    api_version: str
    kind: str
    metadata: V1ObjectMetaType
    spec: V1SelfSubjectAccessReviewSpecType
    status: V1SubjectAccessReviewStatusType
