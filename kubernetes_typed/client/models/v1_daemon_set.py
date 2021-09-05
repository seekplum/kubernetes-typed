# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1DaemonSetDict generated type."""
from typing import TypedDict

from kubernetes_typed.client import V1DaemonSetSpecDict, V1DaemonSetStatusDict, V1ObjectMetaDict

V1DaemonSetDict = TypedDict(
    "V1DaemonSetDict",
    {
        "apiVersion": str,
        "kind": str,
        "metadata": V1ObjectMetaDict,
        "spec": V1DaemonSetSpecDict,
        "status": V1DaemonSetStatusDict,
    },
    total=False,
)