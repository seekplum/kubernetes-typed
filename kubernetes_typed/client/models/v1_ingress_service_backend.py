# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1IngressServiceBackendType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


from .v1_service_backend_port import V1ServiceBackendPortType


@dataclass
class V1IngressServiceBackendType:
    name: str
    port: V1ServiceBackendPortType
