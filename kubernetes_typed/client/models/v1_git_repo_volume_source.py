# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1GitRepoVolumeSourceType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


@dataclass
class V1GitRepoVolumeSourceType:
    directory: str
    repository: str
    revision: str
