# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1ContainerStateType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


from .v1_container_state_running import V1ContainerStateRunningType
from .v1_container_state_terminated import V1ContainerStateTerminatedType
from .v1_container_state_waiting import V1ContainerStateWaitingType


@dataclass
class V1ContainerStateType:
    running: V1ContainerStateRunningType
    terminated: V1ContainerStateTerminatedType
    waiting: V1ContainerStateWaitingType
