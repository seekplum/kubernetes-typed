# Code generated by `typeddictgen`. DO NOT EDIT.
"""V2HPAScalingPolicyType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


@dataclass
class V2HPAScalingPolicyType:
    period_seconds: int
    type: str
    value: int
