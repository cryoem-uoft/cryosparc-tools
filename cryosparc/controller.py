"""
Core base classes and utilities for other cryosparc-tools modules.
"""

# NOTE: This file should only include utilities required only by cryosparc-tools
# CryoSPARC should not depend on anything in this file.
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar, Union

from pydantic import BaseModel

from .models.job_spec import InputSlot, OutputSlot
from .spec import Datafield

# API model
M = TypeVar("M", bound=BaseModel)


class Controller(ABC, Generic[M]):
    """
    Abstract base class for Project, Workspace, Job classes and any other types
    that have underlying Mongo database documents.

    Generic type argument D is a typed dictionary definition for a Mongo
    document.

    :meta private:
    """

    _model: Optional[M] = None

    @property
    def model(self) -> M:
        if not self._model:
            self.refresh()
        assert self._model, "Could not refresh database document"
        return self._model

    @property
    def doc(self) -> Dict[str, Any]:
        warnings.warn(".doc attribute is deprecated. Use .model attribute instead.", DeprecationWarning)
        return self.model.model_dump(by_alias=True)

    @abstractmethod
    def refresh(self):
        # Must be implemented in subclasses
        return self


InputSlotSpec = Union[str, InputSlot, Datafield]
"""
A result slot specification for the slots=... argument when creating inputs.
"""

OutputSlotSpec = Union[str, OutputSlot, Datafield]
"""
A result slot specification for the slots=... argument when creating outputs.
"""

LoadableSlots = Union[Literal["default", "passthrough", "all"], List[str]]
"""Slots groups load for a job input or output."""


def as_input_slot(spec: InputSlotSpec) -> InputSlot:
    if isinstance(spec, str):
        spec, required = (spec[1:], False) if spec[0] == "?" else (spec, True)
        return InputSlot(name=spec, dtype=spec, required=required)
    elif isinstance(spec, dict) and "dtype" in spec and "prefix" in spec:
        name, dtype, required = spec["prefix"], spec["dtype"].split(".").pop(), spec.get("required", True)
        return InputSlot(name=name, dtype=dtype, required=required)
    return spec


def as_output_slot(spec: OutputSlotSpec) -> OutputSlot:
    if isinstance(spec, str):
        return OutputSlot(name=spec, dtype=spec)
    elif isinstance(spec, dict) and "dtype" in spec and "prefix" in spec:
        name, dtype = spec["prefix"], spec["dtype"].split(".").pop()
        return OutputSlot(name=name, dtype=dtype)
    return spec
