# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1ResourceRuleDict generated type."""
from typing import TypedDict, List

V1ResourceRuleDict = TypedDict(
    "V1ResourceRuleDict",
    {
        "apiGroups": List[str],
        "resourceNames": List[str],
        "resources": List[str],
        "verbs": List[str],
    },
    total=False,
)