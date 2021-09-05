# Code generated by `typeddictgen`. DO NOT EDIT.
"""V2beta1HorizontalPodAutoscalerSpecDict generated type."""
from typing import TypedDict, List

from kubernetes_typed.client import V2beta1CrossVersionObjectReferenceDict, V2beta1MetricSpecDict

V2beta1HorizontalPodAutoscalerSpecDict = TypedDict(
    "V2beta1HorizontalPodAutoscalerSpecDict",
    {
        "maxReplicas": int,
        "metrics": List[V2beta1MetricSpecDict],
        "minReplicas": int,
        "scaleTargetRef": V2beta1CrossVersionObjectReferenceDict,
    },
    total=False,
)