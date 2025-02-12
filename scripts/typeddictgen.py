#!/usr/bin/env python3

"""Generating kubernetes model dicts."""

import inspect
import keyword
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader
from kubernetes import client as kubernetes_client

from kubernetes_typed.plugin import (
    ATTRIBUTE_NAME_ATTRIBUTE,
    KUBERNETES_CLIENT_PREFIX,
    NATIVE_TYPES_MAPPING,
    OPENAPI_ATTRIBUTE,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from scripts.generate_utils import PROJECT_DIRECTORY, comment_codegen, format_codegen

DICT_CLIENT_TEMPLATE_DIRECTORY = PROJECT_DIRECTORY / "scripts" / "templates" / "typeddict"
DICT_CLIENT_DIRECTORY = PROJECT_DIRECTORY / "kubernetes_typed" / "client"
DICT_CLIENT_MODELS_DIRECTORY = DICT_CLIENT_DIRECTORY / "models"
CLASS_SUFFIX = "Type"


def _is_python_keyword(s: str) -> bool:
    return s in keyword.kwlist


def _repl_char(match: re.Match) -> str:
    if match.span()[0] == 0:
        return ""
    return "_"


def _remove_special_chars(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_]', _repl_char, s)


class Attribute:
    """Represents parsed state of kubernetes client model attribute."""

    def __init__(
        self,
        name: str,
        class_name: str,
        model_name: str,
        *,
        field_value: Optional[str] = None,
        field_import: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Parse attribute parameters."""
        self.name = name
        self.class_name = class_name
        self.model_name = model_name
        self.direct_import: List[str] = []
        self.typing_import: List[str] = []
        self.model_import: DefaultDict[str, List[str]] = defaultdict(list)

        self.type = self.parse_type(class_name)
        self.field_value = field_value
        for m, imps in (field_import or {}).items():
            self.model_import[m].extend(imps)

    def parse_type(self, class_name: str) -> str:
        """Get attribute type from its class name."""
        # Reference kubernetes.client.api_client.deserialize
        if class_name.startswith("list["):
            self.typing_import.append("List")

            sub_class_name = re.match(r"list\[(.*)\]", class_name).group(1)  # type: ignore

            typ: str = self.parse_type(sub_class_name)

            return "List[{0}]".format(typ)

        if class_name.startswith("dict("):
            self.typing_import.append("Dict")

            key_name = re.match(r"dict\(([^,]*), (.*)\)", class_name).group(1)  # type: ignore
            sub_class_name = re.match(r"dict\(([^,]*), (.*)\)", class_name).group(2)  # type: ignore

            key = self.parse_type(key_name)
            typ = self.parse_type(sub_class_name)

            return "Dict[{0}, {1}]".format(key, typ)

        if NATIVE_TYPES_MAPPING.get(class_name) is not None:
            klass = NATIVE_TYPES_MAPPING[class_name]
            module = klass.__module__

            if module == "builtins":
                return "{0}".format(klass.__qualname__)

            self.direct_import.append(module)
            return "{0}.{1}".format(module, klass.__qualname__)

        klass = getattr(kubernetes_client, class_name, None)

        if klass is None:
            raise NameError("Attribute with missing model: {0}".format(class_name))

        module = klass.__module__.replace("kubernetes.", "kubernetes_typed.")
        typ = "{0}{1}".format(klass.__qualname__, CLASS_SUFFIX)

        # recursive types not supported https://github.com/python/mypy/issues/731
        if typ == self.model_name:
            self.typing_import.append("Any")
            self.typing_import.append("Dict")
            typ = "Dict[Any, Any]"
        else:
            self.model_import[module].append(typ)

        return typ


class Model:  # pylint: disable=too-many-instance-attributes
    """Represents parsed state of kubernetes client model."""

    def __init__(self, class_name: str, klass: object) -> None:
        """Parse model parameters."""
        self.class_name = class_name
        self.klass = klass

        if not filter_models_classes(klass):
            raise NameError("Incompatible module for Models class: {0}".format(klass.__module__))

        self.module_full_name = klass.__module__.replace("kubernetes.", "kubernetes_typed.")
        self.module_name = klass.__module__.rpartition(".")[2]
        self.name = "{0}{1}".format(class_name, CLASS_SUFFIX)

        oapi: Dict[str, str] = getattr(klass, OPENAPI_ATTRIBUTE)
        attrs: Dict[str, str] = getattr(klass, ATTRIBUTE_NAME_ATTRIBUTE)

        self.attributes: List[Attribute] = []

        for name, typ in oapi.items():
            attr_name = attrs[name]
            field_value = None
            field_import = None
            new_attr_name = _remove_special_chars(attr_name)
            is_keyword = _is_python_keyword(new_attr_name)
            if is_keyword or new_attr_name != attr_name:
                field_value = 'field(metadata={{"alias": "{0}"}})'.format(attr_name)
                field_import = {"dataclasses": ["field"]}
            if is_keyword:
                attr_name = f"{new_attr_name}_"
            else:
                attr_name = new_attr_name

            self.attributes.append(
                Attribute(
                    attr_name,
                    typ,
                    self.name,
                    field_value=field_value,
                    field_import=field_import,
                )
            )

        self.direct_import = self.uniq_imports([attr.direct_import for attr in self.attributes])
        self.typing_import = self.uniq_imports([attr.typing_import for attr in self.attributes])
        model_import = {
            module_name: imports
            for attr in self.attributes
            for module_name, imports in attr.model_import.items()
        }
        model_import.setdefault("dataclasses", []).append("dataclass")
        self.model_import = {
            module_name: self.uniq_imports([imports])
            for module_name, imports in model_import.items()
        }

    def uniq_imports(self, imports: List[List[str]]) -> List[str]:
        """Get uniq import for the model."""
        # flatten and uniq and sort
        return sorted({imp for sublist in imports if sublist for imp in sublist if imp and imp != self.name})


def filter_models_classes(klass: object) -> bool:
    """Filter out classes that don't belong to models module."""
    try:
        check = klass.__module__.startswith(KUBERNETES_CLIENT_PREFIX)
    except AttributeError:
        check = False

    return check


def generate_dicts(client_dir: Path, models_dir: Path) -> None:  # pylint: disable=too-many-locals
    """Generate TypedDict for kubernetes models."""
    model_classes = inspect.getmembers(kubernetes_client, filter_models_classes)

    models: List[Model] = []

    for name, klass in model_classes:
        models.append(Model(name, klass))

    loader = FileSystemLoader(searchpath=DICT_CLIENT_TEMPLATE_DIRECTORY)
    library = Environment(loader=loader, autoescape=True)

    template = library.get_template("__init__.py.j2")

    init_definition = template.render(models=models)

    if client_dir.exists():
        shutil.rmtree(client_dir)

    os.makedirs(client_dir)
    with open(client_dir / "__init__.py", "w+", encoding="utf-8") as codegen_file:
        codegen_file.write(init_definition)

    os.makedirs(models_dir)
    with open(models_dir / "__init__.py", "w+", encoding="utf-8") as codegen_file:
        codegen_file.write(init_definition)

    for model in models:
        template = library.get_template("class.py.j2")

        klass_definition = template.render(model=model)

        with open(models_dir / f"{model.module_name}.py", "w+", encoding="utf-8") as codegen_file:
            codegen_file.write(klass_definition)

    comment_codegen(client_dir, "typeddictgen")
    format_codegen(client_dir)


def main() -> None:
    generate_dicts(DICT_CLIENT_DIRECTORY, DICT_CLIENT_MODELS_DIRECTORY)


if __name__ == "__main__":
    main()
