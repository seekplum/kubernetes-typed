# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1PodDict generated type."""
from typing import TypedDict

from kubernetes_typed.client import V1ObjectMetaDict, V1PodSpecDict, V1PodStatusDict

V1PodDict = TypedDict(
    "V1PodDict",
    {
        "apiVersion": str,
        "kind": str,
        "metadata": V1ObjectMetaDict,
        "spec": V1PodSpecDict,
        "status": V1PodStatusDict,
    },
    total=False,
)