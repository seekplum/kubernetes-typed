# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1FlowSchemaStatusType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List

from .v1_flow_schema_condition import V1FlowSchemaConditionType


@dataclass
class V1FlowSchemaStatusType:
    conditions: List[V1FlowSchemaConditionType]