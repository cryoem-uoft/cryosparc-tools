"""
Model registration functions used by API client to determine how to interpret
JSON responses. Used for either cryosparc-tools or cryosparc models.
"""

import re
import warnings
from enum import Enum
from inspect import isclass
from types import ModuleType
from typing import Dict, Iterable, Optional, Type

from pydantic import BaseModel

from .stream import Streamable

FINALIZED: bool = False
REGISTERED_TYPED_DICTS: Dict[str, Type[dict]] = {}
REGISTERED_ENUMS: Dict[str, Type[Enum]] = {}
REGISTERED_MODEL_CLASSES: Dict[str, Type[BaseModel]] = {}
REGISTERED_STREAM_CLASSES: Dict[str, Type[Streamable]] = {}


def finalize():
    """
    Prevent registering additional types. Cannot be called twice.
    """
    global FINALIZED
    check_finalized(False)
    FINALIZED = True


def check_finalized(finalized: bool = True):
    """
    Ensure the register has or hasn't been finalized. This is used in
    special contexts such as cryosparcm icli or Jupyter Notebooks where
    cryosparc-tools may be used alongside an API client.
    """
    assert FINALIZED is finalized, (
        f"Cannot proceed because registry is {'finalized' if FINALIZED else 'not finalized'}. "
        "This likely means that you're using both cryosparc-tools AND the "
        "CryoSPARC API client from client/api_client.py. Please use either "
        "`CryoSPARC` from tools or `APIClient` from cryosparc, but not both."
    )


def register_model(name, model_class: Type[BaseModel]):
    check_finalized(False)
    REGISTERED_MODEL_CLASSES[name] = model_class


def register_typed_dict(name, typed_dict_class: Type[dict]):
    check_finalized(False)
    REGISTERED_TYPED_DICTS[name] = typed_dict_class


def register_enum(name, enum_class: Type[Enum]):
    check_finalized(False)
    REGISTERED_ENUMS[name] = enum_class


def register_model_module(mod: ModuleType):
    for key, val in mod.__dict__.items():
        if not re.match(r"^[A-Z]", key) or not isclass(val):
            continue
        if issubclass(val, BaseModel):
            register_model(key, val)
        if issubclass(val, dict):
            register_typed_dict(key, val)
        if issubclass(val, Enum):
            register_enum(key, val)


def model_for_ref(schema_ref: str) -> Optional[Type]:
    """
    Given a string with format either `#/components/schemas/X` or
    `#/components/schemas/X_Y_`, looks up key X in `REGISTERED_MODEL_CLASSES``,
    and return either X or X[Y] depending on whether the string includes the
    final Y component.

    Returns None if ref is not found.
    """
    components = schema_ref.split("/")
    if len(components) != 4 or components[0] != "#" or components[1] != "components" or components[2] != "schemas":
        warnings.warn(f"Warning: Invalid schema reference {schema_ref}", stacklevel=2)
        return

    schema_name = components[3]
    if "_" in schema_name:  # type var
        generic, var, *_ = schema_name.split("_")
        if generic in REGISTERED_MODEL_CLASSES and var in REGISTERED_MODEL_CLASSES:
            return REGISTERED_MODEL_CLASSES[generic][REGISTERED_MODEL_CLASSES[var]]  # type: ignore
    elif schema_name in REGISTERED_MODEL_CLASSES:
        return REGISTERED_MODEL_CLASSES[schema_name]
    elif schema_name in REGISTERED_TYPED_DICTS:
        return REGISTERED_TYPED_DICTS[schema_name]
    elif schema_name in REGISTERED_ENUMS:
        return REGISTERED_ENUMS[schema_name]

    warnings.warn(f"Warning: Unknown schema reference model {schema_ref}", stacklevel=2)


def is_streamable_mime_type(mime: str):
    return mime in REGISTERED_STREAM_CLASSES


def register_stream_class(stream_class: Type[Streamable]):
    mime = stream_class.media_type
    assert mime not in REGISTERED_STREAM_CLASSES, (
        f"Cannot register {stream_class}; "
        f"stream class with mime-type {mime} is already registered "
        f"({REGISTERED_STREAM_CLASSES[mime]})"
    )
    REGISTERED_STREAM_CLASSES[mime] = stream_class


def get_stream_class(mime: str):
    return REGISTERED_STREAM_CLASSES.get(mime)  # fails if mime-type not defined


def streamable_mime_types():
    return set(REGISTERED_STREAM_CLASSES.keys())


def first_streamable_mime(strs: Iterable[str]) -> Optional[str]:
    mimes = streamable_mime_types() & set(strs)
    return mimes.pop() if len(mimes) > 0 else None
