# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1PodFailurePolicyOnExitCodesRequirementType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List


@dataclass
class V1PodFailurePolicyOnExitCodesRequirementType:
    container_name: str
    operator: str
    values: List[int]
