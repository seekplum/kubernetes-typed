# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1alpha1StorageVersionConditionType generated type."""
# pylint: disable=too-many-instance-attributes
import datetime
from dataclasses import dataclass


@dataclass
class V1alpha1StorageVersionConditionType:
    last_transition_time: datetime.datetime
    message: str
    observed_generation: int
    reason: str
    status: str
    type: str
