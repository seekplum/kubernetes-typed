# Code generated by `typeddictgen`. DO NOT EDIT.
"""ExtensionsV1beta1IngressDict generated type."""
from typing import TypedDict

from kubernetes_typed.client import ExtensionsV1beta1IngressSpecDict, ExtensionsV1beta1IngressStatusDict, V1ObjectMetaDict

ExtensionsV1beta1IngressDict = TypedDict(
    "ExtensionsV1beta1IngressDict",
    {
        "apiVersion": str,
        "kind": str,
        "metadata": V1ObjectMetaDict,
        "spec": ExtensionsV1beta1IngressSpecDict,
        "status": ExtensionsV1beta1IngressStatusDict,
    },
    total=False,
)