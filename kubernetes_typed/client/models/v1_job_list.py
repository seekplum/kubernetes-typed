# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1JobListType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List

from .v1_job import V1JobType
from .v1_list_meta import V1ListMetaType


@dataclass
class V1JobListType:
    api_version: str
    items: List[V1JobType]
    kind: str
    metadata: V1ListMetaType
