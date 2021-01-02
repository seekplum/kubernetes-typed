# Code generated by `stubgen`. DO NOT EDIT.
from kubernetes.client.configuration import Configuration as Configuration
from typing import Any, Optional

class ExtensionsV1beta1IngressSpec:
    openapi_types: Any = ...
    attribute_map: Any = ...
    local_vars_configuration: Any = ...
    discriminator: Any = ...
    def __init__(self, backend: Optional[Any] = ..., rules: Optional[Any] = ..., tls: Optional[Any] = ..., local_vars_configuration: Optional[Any] = ...) -> None: ...
    @property
    def backend(self): ...
    @backend.setter
    def backend(self, backend: Any) -> None: ...
    @property
    def rules(self): ...
    @rules.setter
    def rules(self, rules: Any) -> None: ...
    @property
    def tls(self): ...
    @tls.setter
    def tls(self, tls: Any) -> None: ...
    def to_dict(self): ...
    def to_str(self): ...
    def __eq__(self, other: Any) -> Any: ...
    def __ne__(self, other: Any) -> Any: ...