# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1PodTemplateSpecType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


from .v1_object_meta import V1ObjectMetaType
from .v1_pod_spec import V1PodSpecType


@dataclass
class V1PodTemplateSpecType:
    metadata: V1ObjectMetaType
    spec: V1PodSpecType
