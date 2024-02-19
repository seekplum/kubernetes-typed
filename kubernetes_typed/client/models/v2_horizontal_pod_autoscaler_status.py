# Code generated by `typeddictgen`. DO NOT EDIT.
"""V2HorizontalPodAutoscalerStatusType generated type."""
# pylint: disable=too-many-instance-attributes
import datetime
from dataclasses import dataclass
from typing import List

from .v2_horizontal_pod_autoscaler_condition import V2HorizontalPodAutoscalerConditionType
from .v2_metric_status import V2MetricStatusType


@dataclass
class V2HorizontalPodAutoscalerStatusType:
    conditions: List[V2HorizontalPodAutoscalerConditionType]
    current_metrics: List[V2MetricStatusType]
    current_replicas: int
    desired_replicas: int
    last_scale_time: datetime.datetime
    observed_generation: int