# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1beta1SelfSubjectReviewType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


from .v1_object_meta import V1ObjectMetaType
from .v1beta1_self_subject_review_status import V1beta1SelfSubjectReviewStatusType


@dataclass
class V1beta1SelfSubjectReviewType:
    api_version: str
    kind: str
    metadata: V1ObjectMetaType
    status: V1beta1SelfSubjectReviewStatusType
