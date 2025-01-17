# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1CronJobListType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List

from .v1_cron_job import V1CronJobType
from .v1_list_meta import V1ListMetaType


@dataclass
class V1CronJobListType:
    api_version: str
    items: List[V1CronJobType]
    kind: str
    metadata: V1ListMetaType
