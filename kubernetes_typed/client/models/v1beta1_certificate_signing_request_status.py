# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1beta1CertificateSigningRequestStatusDict generated type."""
from typing import TypedDict, List

from kubernetes_typed.client import V1beta1CertificateSigningRequestConditionDict

V1beta1CertificateSigningRequestStatusDict = TypedDict(
    "V1beta1CertificateSigningRequestStatusDict",
    {
        "certificate": str,
        "conditions": List[V1beta1CertificateSigningRequestConditionDict],
    },
    total=False,
)