# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1alpha1VolumeAttributesClassType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import Dict

from .v1_object_meta import V1ObjectMetaType


@dataclass
class V1alpha1VolumeAttributesClassType:
    api_version: str
    driver_name: str
    kind: str
    metadata: V1ObjectMetaType
    parameters: Dict[str, str]
