# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1beta1NamedRuleWithOperationsType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List


@dataclass
class V1beta1NamedRuleWithOperationsType:
    api_groups: List[str]
    api_versions: List[str]
    operations: List[str]
    resource_names: List[str]
    resources: List[str]
    scope: str