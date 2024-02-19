# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1PersistentVolumeClaimSpecType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List

from .v1_typed_local_object_reference import V1TypedLocalObjectReferenceType
from .v1_typed_object_reference import V1TypedObjectReferenceType
from .v1_volume_resource_requirements import V1VolumeResourceRequirementsType
from .v1_label_selector import V1LabelSelectorType


@dataclass
class V1PersistentVolumeClaimSpecType:
    access_modes: List[str]
    data_source: V1TypedLocalObjectReferenceType
    data_source_ref: V1TypedObjectReferenceType
    resources: V1VolumeResourceRequirementsType
    selector: V1LabelSelectorType
    storage_class_name: str
    volume_attributes_class_name: str
    volume_mode: str
    volume_name: str
