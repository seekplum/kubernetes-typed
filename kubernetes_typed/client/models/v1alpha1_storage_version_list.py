# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1alpha1StorageVersionListType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List

from .v1alpha1_storage_version import V1alpha1StorageVersionType
from .v1_list_meta import V1ListMetaType


@dataclass
class V1alpha1StorageVersionListType:
    api_version: str
    items: List[V1alpha1StorageVersionType]
    kind: str
    metadata: V1ListMetaType
