# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1RoleBindingDict generated type."""
from typing import TypedDict, List

from kubernetes_typed.client import V1ObjectMetaDict, V1RoleRefDict, V1SubjectDict

V1RoleBindingDict = TypedDict(
    "V1RoleBindingDict",
    {
        "apiVersion": str,
        "kind": str,
        "metadata": V1ObjectMetaDict,
        "roleRef": V1RoleRefDict,
        "subjects": List[V1SubjectDict],
    },
    total=False,
)