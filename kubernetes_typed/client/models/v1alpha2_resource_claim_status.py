# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1alpha2ResourceClaimStatusType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List

from .v1alpha2_allocation_result import V1alpha2AllocationResultType
from .v1alpha2_resource_claim_consumer_reference import V1alpha2ResourceClaimConsumerReferenceType


@dataclass
class V1alpha2ResourceClaimStatusType:
    allocation: V1alpha2AllocationResultType
    deallocation_requested: bool
    driver_name: str
    reserved_for: List[V1alpha2ResourceClaimConsumerReferenceType]