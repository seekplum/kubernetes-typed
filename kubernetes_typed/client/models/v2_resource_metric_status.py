# Code generated by `typeddictgen`. DO NOT EDIT.
"""V2ResourceMetricStatusType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


from .v2_metric_value_status import V2MetricValueStatusType


@dataclass
class V2ResourceMetricStatusType:
    current: V2MetricValueStatusType
    name: str