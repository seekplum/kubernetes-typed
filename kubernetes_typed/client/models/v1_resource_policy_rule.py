# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1ResourcePolicyRuleType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List


@dataclass
class V1ResourcePolicyRuleType:
    api_groups: List[str]
    cluster_scope: bool
    namespaces: List[str]
    resources: List[str]
    verbs: List[str]
