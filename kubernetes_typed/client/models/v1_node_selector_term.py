# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1NodeSelectorTermDict generated type."""
from typing import TypedDict, List

from kubernetes_typed.client import V1NodeSelectorRequirementDict

V1NodeSelectorTermDict = TypedDict(
    "V1NodeSelectorTermDict",
    {
        "matchExpressions": List[V1NodeSelectorRequirementDict],
        "matchFields": List[V1NodeSelectorRequirementDict],
    },
    total=False,
)