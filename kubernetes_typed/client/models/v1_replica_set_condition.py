# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1ReplicaSetConditionType generated type."""
# pylint: disable=too-many-instance-attributes
import datetime
from dataclasses import dataclass


@dataclass
class V1ReplicaSetConditionType:
    last_transition_time: datetime.datetime
    message: str
    reason: str
    status: str
    type: str
