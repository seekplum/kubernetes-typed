# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1PolicyRulesWithSubjectsType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List

from .v1_non_resource_policy_rule import V1NonResourcePolicyRuleType
from .v1_resource_policy_rule import V1ResourcePolicyRuleType
from .flowcontrol_v1_subject import FlowcontrolV1SubjectType


@dataclass
class V1PolicyRulesWithSubjectsType:
    non_resource_rules: List[V1NonResourcePolicyRuleType]
    resource_rules: List[V1ResourcePolicyRuleType]
    subjects: List[FlowcontrolV1SubjectType]
