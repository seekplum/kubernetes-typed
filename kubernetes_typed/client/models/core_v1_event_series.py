# Code generated by `typeddictgen`. DO NOT EDIT.
"""CoreV1EventSeriesType generated type."""
# pylint: disable=too-many-instance-attributes
import datetime
from dataclasses import dataclass


@dataclass
class CoreV1EventSeriesType:
    count: int
    last_observed_time: datetime.datetime
