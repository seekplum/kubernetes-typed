# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1PodDisruptionBudgetType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


from .v1_object_meta import V1ObjectMetaType
from .v1_pod_disruption_budget_spec import V1PodDisruptionBudgetSpecType
from .v1_pod_disruption_budget_status import V1PodDisruptionBudgetStatusType


@dataclass
class V1PodDisruptionBudgetType:
    api_version: str
    kind: str
    metadata: V1ObjectMetaType
    spec: V1PodDisruptionBudgetSpecType
    status: V1PodDisruptionBudgetStatusType