# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1alpha2ResourceClaimTemplateListType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List

from .v1alpha2_resource_claim_template import V1alpha2ResourceClaimTemplateType
from .v1_list_meta import V1ListMetaType


@dataclass
class V1alpha2ResourceClaimTemplateListType:
    api_version: str
    items: List[V1alpha2ResourceClaimTemplateType]
    kind: str
    metadata: V1ListMetaType
