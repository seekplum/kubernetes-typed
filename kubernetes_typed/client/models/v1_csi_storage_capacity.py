# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1CSIStorageCapacityType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


from .v1_object_meta import V1ObjectMetaType
from .v1_label_selector import V1LabelSelectorType


@dataclass
class V1CSIStorageCapacityType:
    api_version: str
    capacity: str
    kind: str
    maximum_volume_size: str
    metadata: V1ObjectMetaType
    node_topology: V1LabelSelectorType
    storage_class_name: str
