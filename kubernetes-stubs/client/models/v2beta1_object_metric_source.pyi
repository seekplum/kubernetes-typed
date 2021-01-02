# Code generated by `stubgen`. DO NOT EDIT.
from kubernetes.client.configuration import Configuration as Configuration
from typing import Any, Optional

class V2beta1ObjectMetricSource:
    openapi_types: Any = ...
    attribute_map: Any = ...
    local_vars_configuration: Any = ...
    discriminator: Any = ...
    def __init__(self, average_value: Optional[Any] = ..., metric_name: Optional[Any] = ..., selector: Optional[Any] = ..., target: Optional[Any] = ..., target_value: Optional[Any] = ..., local_vars_configuration: Optional[Any] = ...) -> None: ...
    @property
    def average_value(self): ...
    @average_value.setter
    def average_value(self, average_value: Any) -> None: ...
    @property
    def metric_name(self): ...
    @metric_name.setter
    def metric_name(self, metric_name: Any) -> None: ...
    @property
    def selector(self): ...
    @selector.setter
    def selector(self, selector: Any) -> None: ...
    @property
    def target(self): ...
    @target.setter
    def target(self, target: Any) -> None: ...
    @property
    def target_value(self): ...
    @target_value.setter
    def target_value(self, target_value: Any) -> None: ...
    def to_dict(self): ...
    def to_str(self): ...
    def __eq__(self, other: Any) -> Any: ...
    def __ne__(self, other: Any) -> Any: ...