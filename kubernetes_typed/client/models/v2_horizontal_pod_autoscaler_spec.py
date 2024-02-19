# Code generated by `typeddictgen`. DO NOT EDIT.
"""V2HorizontalPodAutoscalerSpecType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List

from .v2_horizontal_pod_autoscaler_behavior import V2HorizontalPodAutoscalerBehaviorType
from .v2_metric_spec import V2MetricSpecType
from .v2_cross_version_object_reference import V2CrossVersionObjectReferenceType


@dataclass
class V2HorizontalPodAutoscalerSpecType:
    behavior: V2HorizontalPodAutoscalerBehaviorType
    max_replicas: int
    metrics: List[V2MetricSpecType]
    min_replicas: int
    scale_target_ref: V2CrossVersionObjectReferenceType
