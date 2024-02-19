#!/usr/bin/env python3

"""Generating kubernetes model dicts."""

import inspect
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List

from jinja2 import Environment, FileSystemLoader
from kubernetes import client as kubernetes_client

from kubernetes_typed.plugin import (
    ATTRIBUTE_NAME_ATTRIBUTE,
    KUBERNETES_CLIENT_PREFIX,
    NATIVE_TYPES_MAPPING,
    OPENAPI_ATTRIBUTE,
)
from scripts.generate_utils import PROJECT_DIRECTORY, comment_codegen, format_codegen

DICT_CLIENT_TEMPLATE_DIRECTORY = PROJECT_DIRECTORY / "scripts" / "templates" / "typeddict"
DICT_CLIENT_DIRECTORY = PROJECT_DIRECTORY / "kubernetes_typed" / "client"
DICT_CLIENT_MODELS_DIRECTORY = DICT_CLIENT_DIRECTORY / "models"
CLASS_SUFFIX = "Type"


class Attribute:
    """Represents parsed state of kubernetes client model attribute."""

    def __init__(self, name: str, class_name: str, model_name: str) -> None:
        """Parse attribute parameters."""
        self.name = name
        self.class_name = class_name
        self.model_name = model_name
        self.direct_import: List[str] = []
        self.typing_import: List[str] = []
        self.model_import: DefaultDict[str, List[str]] = defaultdict(list)

        self.type = self.parse_type(class_name)

    def parse_type(self, class_name: str) -> str:
        """Get attribute type from its class name."""
        # Reference kubernetes.client.api_client.deserialize
        if class_name.startswith("list["):
            self.typing_import.append("List")

            sub_class_name = re.match(r"list\[(.*)\]", class_name).group(1)  # type: ignore

            typ: str = self.parse_type(sub_class_name)

            return f"List[{typ}]"

        if class_name.startswith("dict("):
            self.typing_import.append("Dict")

            key_name = re.match(r"dict\(([^,]*), (.*)\)", class_name).group(1)  # type: ignore
            sub_class_name = re.match(r"dict\(([^,]*), (.*)\)", class_name).group(2)  # type: ignore

            key = self.parse_type(key_name)
            typ = self.parse_type(sub_class_name)

            return f"Dict[{key}, {typ}]"

        if NATIVE_TYPES_MAPPING.get(class_name) is not None:
            klass = NATIVE_TYPES_MAPPING[class_name]
            module = klass.__module__

            if module == "builtins":
                return klass.__qualname__

            self.direct_import.append(module)
            return f"{module}.{klass.__qualname__}"

        klass = getattr(kubernetes_client, class_name, None)

        if klass is None:
            raise NameError(f"Attribute with missing model: {class_name}")

        typ = f"{klass.__qualname__}{CLASS_SUFFIX}"
        self.model_import[klass.__module__.rsplit(".", 1)[-1]].append(typ)

        # recursive types not supported https://github.com/python/mypy/issues/731
        if typ == self.model_name:
            self.typing_import.append("Any")
            self.typing_import.append("Dict")
            typ = "Dict[Any, Any]"

        return typ


class Model:  # pylint: disable=too-many-instance-attributes
    """Represents parsed state of kubernetes client model."""

    def __init__(self, class_name: str, klass: object) -> None:
        """Parse model parameters."""
        self.class_name = class_name
        self.klass = klass

        if not filter_models_classes(klass):
            raise NameError(f"Incompatible module for Models class: {klass.__module__}")

        self.module_full_name = klass.__module__.replace("kubernetes.", "kubernetes_typed.")
        self.module_name = klass.__module__.rpartition(".")[2]
        self.name = f"{class_name}{CLASS_SUFFIX}"

        oapi: Dict[str, str] = getattr(klass, OPENAPI_ATTRIBUTE)

        self.attributes: List[Attribute] = []

        for name, typ in oapi.items():
            self.attributes.append(Attribute(name, typ, self.name))

        self.direct_import = self.uniq_imports([attr.direct_import for attr in self.attributes])
        self.typing_import = self.uniq_imports([attr.typing_import for attr in self.attributes])
        self.model_import = {
            module_name: new_imports
            for attr in self.attributes
            for module_name, imports in attr.model_import.items()
            if (new_imports := self.uniq_imports([imports]))
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
