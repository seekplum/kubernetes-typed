# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1beta1VolumeAttachmentSpecDict generated type."""
from typing import TypedDict

from kubernetes_typed.client import V1beta1VolumeAttachmentSourceDict

V1beta1VolumeAttachmentSpecDict = TypedDict(
    "V1beta1VolumeAttachmentSpecDict",
    {
        "attacher": str,
        "nodeName": str,
        "source": V1beta1VolumeAttachmentSourceDict,
    },
    total=False,
)