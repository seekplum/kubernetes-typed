# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1alpha1IPAddressType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


from .v1_object_meta import V1ObjectMetaType
from .v1alpha1_ip_address_spec import V1alpha1IPAddressSpecType


@dataclass
class V1alpha1IPAddressType:
    api_version: str
    kind: str
    metadata: V1ObjectMetaType
    spec: V1alpha1IPAddressSpecType
