# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1WebhookConversionType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass
from typing import List

from .apiextensions_v1_webhook_client_config import ApiextensionsV1WebhookClientConfigType


@dataclass
class V1WebhookConversionType:
    client_config: ApiextensionsV1WebhookClientConfigType
    conversion_review_versions: List[str]
