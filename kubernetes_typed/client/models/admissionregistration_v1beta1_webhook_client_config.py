# Code generated by `typeddictgen`. DO NOT EDIT.
"""AdmissionregistrationV1beta1WebhookClientConfigDict generated type."""
from typing import TypedDict

from kubernetes_typed.client import AdmissionregistrationV1beta1ServiceReferenceDict

AdmissionregistrationV1beta1WebhookClientConfigDict = TypedDict(
    "AdmissionregistrationV1beta1WebhookClientConfigDict",
    {
        "caBundle": str,
        "service": AdmissionregistrationV1beta1ServiceReferenceDict,
        "url": str,
    },
    total=False,
)