# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1beta1CustomResourceSubresourcesDict generated type."""
from typing import TypedDict

from kubernetes_typed.client import V1beta1CustomResourceSubresourceScaleDict

V1beta1CustomResourceSubresourcesDict = TypedDict(
    "V1beta1CustomResourceSubresourcesDict",
    {
        "scale": V1beta1CustomResourceSubresourceScaleDict,
        "status": object,
    },
    total=False,
)