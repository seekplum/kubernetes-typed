# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1alpha2ResourceClaimSpecType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


from .v1alpha2_resource_claim_parameters_reference import V1alpha2ResourceClaimParametersReferenceType


@dataclass
class V1alpha2ResourceClaimSpecType:
    allocation_mode: str
    parameters_ref: V1alpha2ResourceClaimParametersReferenceType
    resource_class_name: str