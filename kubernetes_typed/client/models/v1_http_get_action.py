# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1HTTPGetActionType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List

from .v1_http_header import V1HTTPHeaderType


@dataclass
class V1HTTPGetActionType:
    host: str
    http_headers: List[V1HTTPHeaderType]
    path: str
    port: object
    scheme: str
