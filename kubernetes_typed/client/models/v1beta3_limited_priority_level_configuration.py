# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1beta3LimitedPriorityLevelConfigurationType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


from .v1beta3_limit_response import V1beta3LimitResponseType


@dataclass
class V1beta3LimitedPriorityLevelConfigurationType:
    borrowing_limit_percent: int
    lendable_percent: int
    limit_response: V1beta3LimitResponseType
    nominal_concurrency_shares: int
