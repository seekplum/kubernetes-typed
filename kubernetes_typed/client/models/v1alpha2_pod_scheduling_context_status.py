# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1alpha2PodSchedulingContextStatusType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List

from .v1alpha2_resource_claim_scheduling_status import V1alpha2ResourceClaimSchedulingStatusType


@dataclass
class V1alpha2PodSchedulingContextStatusType:
    resource_claims: List[V1alpha2ResourceClaimSchedulingStatusType]
