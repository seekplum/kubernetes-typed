# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1ContainerStateDict generated type."""
from typing import TypedDict

from kubernetes_typed.client import V1ContainerStateRunningDict, V1ContainerStateTerminatedDict, V1ContainerStateWaitingDict

V1ContainerStateDict = TypedDict(
    "V1ContainerStateDict",
    {
        "running": V1ContainerStateRunningDict,
        "terminated": V1ContainerStateTerminatedDict,
        "waiting": V1ContainerStateWaitingDict,
    },
    total=False,
)