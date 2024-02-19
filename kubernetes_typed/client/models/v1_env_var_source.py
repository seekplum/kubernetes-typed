# Code generated by `typeddictgen`. DO NOT EDIT.
"""V1EnvVarSourceType generated type."""
# pylint: disable=too-many-instance-attributes
from dataclasses import dataclass


from .v1_config_map_key_selector import V1ConfigMapKeySelectorType
from .v1_object_field_selector import V1ObjectFieldSelectorType
from .v1_resource_field_selector import V1ResourceFieldSelectorType
from .v1_secret_key_selector import V1SecretKeySelectorType


@dataclass
class V1EnvVarSourceType:
    config_map_key_ref: V1ConfigMapKeySelectorType
    field_ref: V1ObjectFieldSelectorType
    resource_field_ref: V1ResourceFieldSelectorType
    secret_key_ref: V1SecretKeySelectorType
