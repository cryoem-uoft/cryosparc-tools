"""
Definitions for various error classes raised by cryosparc-tools functions
"""

from typing import Any, List

from typing_extensions import TypedDict

from .spec import Datafield, Datatype, SlotSpec


class DatasetLoadError(Exception):
    """Exception type raised when a dataset cannot be loaded"""

    pass


class CommandError(Exception):
    """
    Raised by failed request to a CryoSPARC command server.
    """

    code: int
    data: Any

    def __init__(self, reason: str, *args: object, url: str = "", code: int = 500, data: Any = None) -> None:
        msg = f"*** ({url}, code {code}) {reason}"
        super().__init__(msg, *args)
        self.code = code
        self.data = data


class SlotsValidation(TypedDict):
    """
    Data from validation error when specifying external result input/output slots.

    :meta private:
    """

    type: Datatype
    valid: List[SlotSpec]
    invalid: List[Datafield]
    valid_dtypes: List[str]


class InvalidSlotsError(ValueError):
    """
    Raised by functions that accept slots arguments when CryoSPARC reports that
    given slots are not valid.
    """

    def __init__(self, caller: str, validation: SlotsValidation):
        type = validation["type"]
        valid_slots = validation["valid"]
        invalid_slots = validation["invalid"]
        valid_dtypes = validation["valid_dtypes"]
        msg = "\n".join(
            [
                f"Unknown {type} slot dtype(s): {', '.join(s['dtype'] for s in invalid_slots)}. "
                "Only the following slot dtypes are valid:",
                "",
            ]
            + [f" - {t}" for t in valid_dtypes]
            + [
                "",
                "If adding a dynamic result such as alignments_class_#, specify a "
                "slots=... argument with a full data field specification:",
                "",
                f"    {caller}(... , slots=[",
                "        ...",
            ]
            + [f"        {repr(s)}," for s in valid_slots]
            + [
                "        {'dtype': '<INSERT HERE>', 'prefix': '%s', 'required': True}," % s["dtype"]
                for s in invalid_slots
            ]
            + ["        ...", "    ])"]
        )

        return super().__init__(msg)
