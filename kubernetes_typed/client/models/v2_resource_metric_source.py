# Code generated by `typeddictgen`. DO NOT EDIT.
"""V2ResourceMetricSourceType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


from .v2_metric_target import V2MetricTargetType


@dataclass
class V2ResourceMetricSourceType:
    name: str
    target: V2MetricTargetType