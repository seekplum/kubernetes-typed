# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1beta1TypeCheckingType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List

from .v1beta1_expression_warning import V1beta1ExpressionWarningType


@dataclass
class V1beta1TypeCheckingType:
    expression_warnings: List[V1beta1ExpressionWarningType]