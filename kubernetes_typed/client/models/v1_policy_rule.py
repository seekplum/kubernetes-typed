# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1PolicyRuleType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List


@dataclass
class V1PolicyRuleType:
    api_groups: List[str]
    non_resource_ur_ls: List[str]
    resource_names: List[str]
    resources: List[str]
    verbs: List[str]
