# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1PersistentVolumeClaimConditionType generated type."""
# pylint: disable=too-many-instance-attributes
import datetime
from dataclasses import dataclass


@dataclass
class V1PersistentVolumeClaimConditionType:
    last_probe_time: datetime.datetime
    last_transition_time: datetime.datetime
    message: str
    reason: str
    status: str
    type: str
