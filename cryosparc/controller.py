"""
Core base classes and utilities for other cryosparc-tools modules.
"""

# NOTE: This file should only include utilities required only by cryosparc-tools
# CryoSPARC should not depend on anything in this file.
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, TypeVar, Union

from pydantic import BaseModel

from .models.job_spec import InputSlot, OutputSlot
from .spec import SlotSpec

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
        """
        Representation of entitity data. Contents may change in CryoSPARC
        over time, use use :py:meth:`refresh` to update.
        """
        if not self._model:
            self.refresh()
        assert self._model, "Could not refresh database document"
        return self._model

    @model.setter
    def model(self, model: M):
        self._model = model

    @model.deleter
    def model(self):
        self._model = None

    @property
    def doc(self) -> Dict[str, Any]:
        warnings.warn(".doc attribute is deprecated. Use .model attribute instead.", DeprecationWarning, stacklevel=2)
        return self.model.model_dump(by_alias=True)

    @abstractmethod
    def refresh(self):
        # Must be implemented in subclasses
        return self


def as_input_slot(spec: Union[SlotSpec, InputSlot]) -> InputSlot:
    if isinstance(spec, str):
        spec, required = (spec[1:], False) if spec[0] == "?" else (spec, True)
        return InputSlot(name=spec, dtype=spec, required=required)
    elif isinstance(spec, dict) and "dtype" in spec:
        dtype = spec["dtype"]
        name = spec.get("name") or spec.get("prefix") or dtype
        required = spec.get("required", True)
        return InputSlot(name=name, dtype=dtype, required=required)
    return spec


def as_output_slot(spec: Union[SlotSpec, OutputSlot]) -> OutputSlot:
    if isinstance(spec, str):
        return OutputSlot(name=spec, dtype=spec)
    elif isinstance(spec, dict) and "dtype" in spec:
        dtype = spec["dtype"]
        name = spec.get("name") or spec.get("prefix") or dtype
        return OutputSlot(name=name, dtype=dtype)
    return spec
