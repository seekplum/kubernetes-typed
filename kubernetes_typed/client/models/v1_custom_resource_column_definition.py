# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1CustomResourceColumnDefinitionType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


@dataclass
class V1CustomResourceColumnDefinitionType:
    description: str
    format: str
    json_path: str
    name: str
    priority: int
    type: str
