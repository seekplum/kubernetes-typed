# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1ValidationRuleType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


@dataclass
class V1ValidationRuleType:
    field_path: str
    message: str
    message_expression: str
    optional_old_self: bool
    reason: str
    rule: str