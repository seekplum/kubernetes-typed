# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1IngressBackendType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


from .v1_typed_local_object_reference import V1TypedLocalObjectReferenceType
from .v1_ingress_service_backend import V1IngressServiceBackendType


@dataclass
class V1IngressBackendType:
    resource: V1TypedLocalObjectReferenceType
    service: V1IngressServiceBackendType
