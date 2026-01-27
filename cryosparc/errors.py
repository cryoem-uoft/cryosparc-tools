"""
Definitions for various error classes raised by cryosparc-tools functions
"""

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from httpx import Response


class DatasetLoadError(Exception):
    """Exception type raised when a dataset cannot be loaded"""

    pass


class APIError(ValueError):
    """
    Raised by failed request to a CryoSPARC API server.
    """

    code: int
    data: Any = None

    def __init__(
        self,
        reason: str,
        *args: object,
        res: "Response",
        data: Any = None,  # must be JSON-encodable if provided
    ) -> None:
        msg = f"*** [API] ({res.request.method} {res.url}, code {res.status_code}) {reason}"
        super().__init__(msg, *args)
        self.code = res.status_code
        if data is not None:
            self.data = data
        elif res.headers.get("Content-Type") == "application/json":
            self.data = res.json()

    def __str__(self):
        s = super().__str__()
        if self.data:
            s += "\nResponse data:\n"
            s += json.dumps(self.data, indent=4)
        return s


class ExternalJobError(Exception):
    """
    Raised during external job lifecycle failures
    """

    pass
