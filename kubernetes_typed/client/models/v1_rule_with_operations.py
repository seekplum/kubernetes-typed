# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1RuleWithOperationsDict generated type."""
from typing import TypedDict, List

V1RuleWithOperationsDict = TypedDict(
    "V1RuleWithOperationsDict",
    {
        "apiGroups": List[str],
        "apiVersions": List[str],
        "operations": List[str],
        "resources": List[str],
        "scope": str,
    },
    total=False,
)