# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1CustomResourceSubresourceScaleType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


@dataclass
class V1CustomResourceSubresourceScaleType:
    label_selector_path: str
    spec_replicas_path: str
    status_replicas_path: str
