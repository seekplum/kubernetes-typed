# Code generated by `stubgen`. DO NOT EDIT.
from kubernetes.client.configuration import Configuration as Configuration
from typing import Any, Optional

class V1APIResource:
    openapi_types: Any = ...
    attribute_map: Any = ...
    local_vars_configuration: Any = ...
    discriminator: Any = ...
    def __init__(self, categories: Optional[Any] = ..., group: Optional[Any] = ..., kind: Optional[Any] = ..., name: Optional[Any] = ..., namespaced: Optional[Any] = ..., short_names: Optional[Any] = ..., singular_name: Optional[Any] = ..., storage_version_hash: Optional[Any] = ..., verbs: Optional[Any] = ..., version: Optional[Any] = ..., local_vars_configuration: Optional[Any] = ...) -> None: ...
    @property
    def categories(self): ...
    @categories.setter
    def categories(self, categories: Any) -> None: ...
    @property
    def group(self): ...
    @group.setter
    def group(self, group: Any) -> None: ...
    @property
    def kind(self): ...
    @kind.setter
    def kind(self, kind: Any) -> None: ...
    @property
    def name(self): ...
    @name.setter
    def name(self, name: Any) -> None: ...
    @property
    def namespaced(self): ...
    @namespaced.setter
    def namespaced(self, namespaced: Any) -> None: ...
    @property
    def short_names(self): ...
    @short_names.setter
    def short_names(self, short_names: Any) -> None: ...
    @property
    def singular_name(self): ...
    @singular_name.setter
    def singular_name(self, singular_name: Any) -> None: ...
    @property
    def storage_version_hash(self): ...
    @storage_version_hash.setter
    def storage_version_hash(self, storage_version_hash: Any) -> None: ...
    @property
    def verbs(self): ...
    @verbs.setter
    def verbs(self, verbs: Any) -> None: ...
    @property
    def version(self): ...
    @version.setter
    def version(self, version: Any) -> None: ...
    def to_dict(self): ...
    def to_str(self): ...
    def __eq__(self, other: Any) -> Any: ...
    def __ne__(self, other: Any) -> Any: ...