# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1ResourceRuleType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List


@dataclass
class V1ResourceRuleType:
    api_groups: List[str]
    resource_names: List[str]
    resources: List[str]
    verbs: List[str]
