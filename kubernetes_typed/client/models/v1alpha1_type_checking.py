# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1alpha1TypeCheckingType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List

from .v1alpha1_expression_warning import V1alpha1ExpressionWarningType


@dataclass
class V1alpha1TypeCheckingType:
    expression_warnings: List[V1alpha1ExpressionWarningType]
