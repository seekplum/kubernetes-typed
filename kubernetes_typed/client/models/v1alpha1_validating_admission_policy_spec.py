# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1alpha1ValidatingAdmissionPolicySpecType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List

from .v1alpha1_audit_annotation import V1alpha1AuditAnnotationType
from .v1alpha1_match_condition import V1alpha1MatchConditionType
from .v1alpha1_match_resources import V1alpha1MatchResourcesType
from .v1alpha1_param_kind import V1alpha1ParamKindType
from .v1alpha1_validation import V1alpha1ValidationType
from .v1alpha1_variable import V1alpha1VariableType


@dataclass
class V1alpha1ValidatingAdmissionPolicySpecType:
    audit_annotations: List[V1alpha1AuditAnnotationType]
    failure_policy: str
    match_conditions: List[V1alpha1MatchConditionType]
    match_constraints: V1alpha1MatchResourcesType
    param_kind: V1alpha1ParamKindType
    validations: List[V1alpha1ValidationType]
    variables: List[V1alpha1VariableType]
