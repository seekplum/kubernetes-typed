# Code generated by `typeddictgen`. DO NOT EDIT.
"""AdmissionregistrationV1WebhookClientConfigType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


from .admissionregistration_v1_service_reference import AdmissionregistrationV1ServiceReferenceType


@dataclass
class AdmissionregistrationV1WebhookClientConfigType:
    ca_bundle: str
    service: AdmissionregistrationV1ServiceReferenceType
    url: str
