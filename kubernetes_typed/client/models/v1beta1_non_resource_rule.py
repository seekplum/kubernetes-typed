# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1beta1NonResourceRuleDict generated type."""
from typing import TypedDict, List

V1beta1NonResourceRuleDict = TypedDict(
    "V1beta1NonResourceRuleDict",
    {
        "nonResourceURLs": List[str],
        "verbs": List[str],
    },
    total=False,
)