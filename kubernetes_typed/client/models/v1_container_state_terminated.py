# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1ContainerStateTerminatedType generated type."""
# pylint: disable=too-many-instance-attributes
import datetime
from dataclasses import dataclass


@dataclass
class V1ContainerStateTerminatedType:
    container_id: str
    exit_code: int
    finished_at: datetime.datetime
    message: str
    reason: str
    signal: int
    started_at: datetime.datetime
