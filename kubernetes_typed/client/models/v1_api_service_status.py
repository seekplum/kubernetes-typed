# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1APIServiceStatusDict generated type."""
from typing import TypedDict, List

from kubernetes_typed.client import V1APIServiceConditionDict

V1APIServiceStatusDict = TypedDict(
    "V1APIServiceStatusDict",
    {
        "conditions": List[V1APIServiceConditionDict],
    },
    total=False,
)