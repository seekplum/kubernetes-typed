# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1NodeConfigSourceType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


from .v1_config_map_node_config_source import V1ConfigMapNodeConfigSourceType


@dataclass
class V1NodeConfigSourceType:
    config_map: V1ConfigMapNodeConfigSourceType
