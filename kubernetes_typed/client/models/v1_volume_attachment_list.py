# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1VolumeAttachmentListType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List

from .v1_volume_attachment import V1VolumeAttachmentType
from .v1_list_meta import V1ListMetaType


@dataclass
class V1VolumeAttachmentListType:
    api_version: str
    items: List[V1VolumeAttachmentType]
    kind: str
    metadata: V1ListMetaType
