# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1CustomResourceConversionType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


from .v1_webhook_conversion import V1WebhookConversionType


@dataclass
class V1CustomResourceConversionType:
    strategy: str
    webhook: V1WebhookConversionType
